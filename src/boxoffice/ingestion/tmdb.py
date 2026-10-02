"""Client TMDB avec cache disque : un fichier JSON par film, relançable sans appel redondant."""
import argparse
import json
import logging
import os
import random
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

import requests

from boxoffice import config
from boxoffice.ingestion import movielens

log = logging.getLogger(__name__)

NOT_FOUND_LOG = "_not_found.jsonl"
MAX_ATTEMPTS = 6

REFERENCE_ENDPOINTS = {
    "genres": "genre/movie/list",
    "certifications": "certification/movie/list",
    "countries": "configuration/countries",
    "languages": "configuration/languages",
    "jobs": "configuration/jobs",
    "configuration": "configuration",
}


class TmdbError(Exception):
    """Erreur transitoire : le film sera retenté au prochain run."""


class NotFound(Exception):
    """404 : la ressource n'existe pas (ou plus) chez TMDB."""


class RateLimited(Exception):
    """429 persistant malgré les attentes."""


class InvalidApiKey(Exception):
    """401 : clé absente, invalide ou révoquée."""


def get_json(session: requests.Session, path: str, params: dict, api_key: str,
             sleep: Callable[[float], None] = time.sleep) -> dict:
    url = f"{config.TMDB_BASE_URL}/{path}"
    params = {**params, "api_key": api_key}
    wait, last_error = 0.0, None
    for attempt in range(MAX_ATTEMPTS):
        if wait:
            sleep(wait)
        try:
            resp = session.get(url, params=params, timeout=30)
        except requests.RequestException as exc:
            last_error, wait = type(exc).__name__, 2 ** attempt
            log.warning("%s : erreur réseau %s, nouvel essai dans %.0f s", path, last_error, wait)
            continue

        status = resp.status_code
        if status == 200:
            return resp.json()
        if status == 404:
            raise NotFound(path)
        if status == 401:
            raise InvalidApiKey("TMDB refuse la clé (401) : vérifier TMDB_API_KEY dans .env")
        if status == 429 or status >= 500:
            last_error = status
            wait = _retry_after(resp, default=2 ** attempt) if status == 429 else 2 ** attempt
            log.warning("%s : HTTP %s, nouvel essai dans %.0f s", path, status, wait)
            continue
        raise TmdbError(f"{path} : HTTP {status} inattendu")

    if last_error == 429:
        raise RateLimited(f"quota TMDB toujours dépassé après {MAX_ATTEMPTS} essais ({path})")
    raise TmdbError(f"{path} : abandon après {MAX_ATTEMPTS} essais ({last_error})")


def _retry_after(resp: requests.Response, default: float) -> float:
    try:
        return min(float(resp.headers["Retry-After"]), 60.0)
    except (KeyError, ValueError):
        return default


def cache_path(cache_dir: Path, tmdb_id: int) -> Path:
    return cache_dir / f"{tmdb_id}.json"


def _write_json_atomic(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def load_not_found(cache_dir: Path) -> set[int]:
    log_file = cache_dir / NOT_FOUND_LOG
    if not log_file.exists():
        return set()
    ids = set()
    for line in log_file.read_text(encoding="utf-8").splitlines():
        try:
            ids.add(json.loads(line)["tmdb_id"])
        except (json.JSONDecodeError, KeyError):
            continue
    return ids


def _record_not_found(cache_dir: Path, tmdb_id: int) -> None:
    entry = {"tmdb_id": tmdb_id, "status": 404,
             "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    with (cache_dir / NOT_FOUND_LOG).open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")


def build_cache(tmdb_ids: Iterable[int], cache_dir: Path, append: Iterable[str],
                api_key: str | None = None, session: requests.Session | None = None,
                sleep: Callable[[float], None] = time.sleep) -> Counter:
    """Télécharge les films absents du cache. Retourne le bilan du run."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    api_key = api_key or config.tmdb_api_key()
    append = tuple(append)
    ids = list(dict.fromkeys(tmdb_ids))
    not_found = load_not_found(cache_dir)
    todo = [i for i in ids if i not in not_found and not cache_path(cache_dir, i).exists()]
    stats = Counter(deja_traites=len(ids) - len(todo))

    params = {"language": config.TMDB_LANGUAGE, "append_to_response": ",".join(append)}
    if "images" in append:
        params["include_image_language"] = "en,null"

    log.info("%d films à demander, %d déjà traités", len(todo), stats["deja_traites"])
    own_session = session is None
    session = session or requests.Session()
    start = time.monotonic()
    try:
        for n, tmdb_id in enumerate(todo, 1):
            try:
                data = get_json(session, f"movie/{tmdb_id}", params, api_key, sleep)
            except NotFound:
                _record_not_found(cache_dir, tmdb_id)
                stats["introuvables_404"] += 1
            except TmdbError as exc:
                log.warning("%s", exc)
                stats["erreurs_transitoires"] += 1
            else:
                _write_json_atomic(cache_path(cache_dir, tmdb_id), data)
                stats["telecharges"] += 1
            if n % 100 == 0 or n == len(todo):
                rate = n / max(time.monotonic() - start, 1e-9)
                log.info("%d/%d (%.1f films/s) %s", n, len(todo), rate, dict(stats))
    finally:
        if own_session:
            session.close()
    return stats


def fetch_reference(api_key: str | None = None, force: bool = False) -> None:
    """Télécharge une fois les listes de référence (genres, certifications, pays, langues…)."""
    api_key = api_key or config.tmdb_api_key()
    config.TMDB_REFERENCE.mkdir(parents=True, exist_ok=True)
    with requests.Session() as session:
        for name, path in REFERENCE_ENDPOINTS.items():
            target = config.TMDB_REFERENCE / f"{name}.json"
            if force or not target.exists():
                _write_json_atomic(target, get_json(session, path, {"language": config.TMDB_LANGUAGE}, api_key))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Construit le cache TMDB (un JSON par film).")
    parser.add_argument("--explore", action="store_true",
                        help="toutes les sous-ressources, dans un cache séparé (exige --sample ou --ids)")
    parser.add_argument("--sample", type=int, help="tirage aléatoire de N films de links.csv")
    parser.add_argument("--ids", type=int, nargs="+", help="identifiants TMDB précis")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.explore and not (args.sample or args.ids):
        parser.error("--explore exige --sample ou --ids")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    append = config.TMDB_APPEND_ALL if args.explore else config.TMDB_APPEND
    cache_dir = config.TMDB_EXPLORE_CACHE if args.explore else config.TMDB_CACHE
    ids = args.ids or movielens.tmdb_ids()
    if args.sample:
        ids = sorted(random.Random(args.seed).sample(ids, min(args.sample, len(ids))))

    try:
        fetch_reference()
        stats = build_cache(ids, cache_dir, append)
    except (InvalidApiKey, RateLimited) as exc:
        log.error("Arrêt : %s. Le cache déjà construit est conservé, relancer plus tard.", exc)
        return 1
    log.info("Terminé : %s", dict(stats))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
