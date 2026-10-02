"""IMDb : téléchargement de title.basics et title.ratings, puis extraction des films retenus."""
import argparse
import csv
import json
import logging
import os

import pandas as pd
import requests

from boxoffice import config
from boxoffice.ingestion import movielens, tmdb

log = logging.getLogger(__name__)

BASICS = "title.basics.tsv.gz"
RATINGS = "title.ratings.tsv.gz"


def download(force: bool = False) -> None:
    config.IMDB_RAW.mkdir(parents=True, exist_ok=True)
    for name in (BASICS, RATINGS):
        target = config.IMDB_RAW / name
        if target.exists() and not force:
            continue
        log.info("Téléchargement de %s", name)
        part = target.with_name(target.name + ".part")
        with requests.get(f"{config.IMDB_URL}/{name}", stream=True, timeout=120) as resp:
            resp.raise_for_status()
            with part.open("wb") as fh:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    fh.write(chunk)
        os.replace(part, target)


def imdb_ids() -> set[str]:
    """tconst des films retenus : ceux de links.csv et ceux que TMDB leur associe."""
    films = movielens.selected_movies()
    ids = set("tt" + films["imdbId"].dropna().str.zfill(7))
    for tmdb_id in films["tmdbId"].dropna().unique():
        path = tmdb.cache_path(config.TMDB_CACHE, tmdb_id)
        if path.exists():
            imdb_id = json.loads(path.read_text(encoding="utf-8")).get("imdb_id")
            if imdb_id:
                ids.add(imdb_id)
    return ids


def _read_filtered(name: str, keep: set[str], **kwargs) -> pd.DataFrame:
    chunks = pd.read_csv(config.IMDB_RAW / name, sep="\t", dtype=str, quoting=csv.QUOTE_NONE,
                         keep_default_na=False, na_values=["\\N"], chunksize=500_000, **kwargs)
    return pd.concat(chunk[chunk["tconst"].isin(keep)] for chunk in chunks)


def build(force: bool = False) -> pd.DataFrame:
    """Table IMDb des films MovieLens : une ligne par tconst (informations de base + note)."""
    if config.IMDB_TABLE.exists() and not force:
        return pd.read_parquet(config.IMDB_TABLE)
    download()
    keep = imdb_ids()
    basics = _read_filtered(BASICS, keep, usecols=["tconst", "titleType", "primaryTitle", "originalTitle",
                                                   "isAdult", "startYear", "runtimeMinutes", "genres"])
    ratings = _read_filtered(RATINGS, keep)
    df = basics.merge(ratings, on="tconst", how="left")
    for col in ("isAdult", "startYear", "runtimeMinutes", "numVotes"):
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    df["averageRating"] = pd.to_numeric(df["averageRating"], errors="coerce")

    config.IMDB_TABLE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(config.IMDB_TABLE, index=False)
    log.info("IMDb : %d titres retrouvés sur %d identifiants demandés", len(df), len(keep))
    return df


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extrait d'IMDb les films MovieLens.")
    parser.add_argument("--force", action="store_true", help="reconstruit la table même si elle existe")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    build(force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
