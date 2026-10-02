"""Référentiel commun des films : une ligne par film retenu avec ses identifiants TMDB et IMDb."""
import json
import logging

import numpy as np
import pandas as pd

from boxoffice import config
from boxoffice.ingestion import imdb, movielens, tmdb

log = logging.getLogger(__name__)

YEAR_IN_TITLE = r"\((\d{4})\)\s*$"
TMDB_COLUMNS = ["tmdbId", "tmdb_status", "tmdb_title", "tmdb_year", "tmdb_imdb_id", "tmdb_budget_revenue"]


def _tmdb_summary(tmdb_ids) -> pd.DataFrame:
    not_found = tmdb.load_not_found(config.TMDB_CACHE)
    rows = []
    for tmdb_id in tmdb_ids:
        path = tmdb.cache_path(config.TMDB_CACHE, tmdb_id)
        if not path.exists():
            rows.append({"tmdbId": tmdb_id, "tmdb_status": "404" if tmdb_id in not_found else "non_telecharge"})
            continue
        movie = json.loads(path.read_text(encoding="utf-8"))
        date = movie.get("release_date") or ""
        rows.append({"tmdbId": tmdb_id, "tmdb_status": "ok", "tmdb_title": movie.get("title"),
                     "tmdb_year": int(date[:4]) if date else None, "tmdb_imdb_id": movie.get("imdb_id") or None,
                     "tmdb_budget_revenue": (movie.get("budget") or 0) > 0 and (movie.get("revenue") or 0) > 0})
    return pd.DataFrame(rows, columns=TMDB_COLUMNS).astype({"tmdbId": "Int64", "tmdb_year": "Int64"})


def build() -> pd.DataFrame:
    ref = movielens.selected_movies().rename(columns={"title": "ml_title", "genres": "ml_genres"})
    ref["ml_year"] = ref["ml_title"].str.extract(YEAR_IN_TITLE)[0].astype("Int64")
    ref["imdb_id_links"] = "tt" + ref.pop("imdbId").str.zfill(7)

    ref = ref.merge(_tmdb_summary(ref["tmdbId"].dropna().unique()), on="tmdbId", how="left")
    ref["tmdb_status"] = ref["tmdb_status"].fillna("sans_tmdbId")
    ref["tmdb_budget_revenue"] = ref["tmdb_budget_revenue"].fillna(False).astype(bool)

    titles = imdb.build()[["tconst", "primaryTitle", "titleType", "startYear"]].rename(
        columns={"tconst": "imdb_id", "primaryTitle": "imdb_title", "titleType": "imdb_type", "startYear": "imdb_year"})
    known = set(titles["imdb_id"])
    from_links = ref["imdb_id_links"].isin(known)
    from_tmdb = ~from_links & ref["tmdb_imdb_id"].isin(known)
    ref["imdb_id"] = ref["imdb_id_links"].where(from_links, ref["tmdb_imdb_id"].where(from_tmdb))
    ref["imdb_source"] = np.select([from_links, from_tmdb], ["links", "tmdb"], default="absent")
    ref = ref.merge(titles, on="imdb_id", how="left")
    ref["imdb_found"] = ref["imdb_title"].notna()

    ref["in_common_list"] = ref["tmdb_status"].eq("ok") & ref["imdb_found"]

    ref["flag_imdb_mismatch"] = ref["tmdb_imdb_id"].notna() & (ref["tmdb_imdb_id"] != ref["imdb_id_links"])
    ref["flag_year_gap_tmdb"] = ((ref["tmdb_year"] - ref["ml_year"]).abs() > 1).fillna(False).astype(bool)
    ref["flag_year_gap_imdb"] = ((ref["imdb_year"] - ref["ml_year"]).abs() > 1).fillna(False).astype(bool)
    ref["flag_duplicate_tmdbId"] = ref["tmdbId"].notna() & ref["tmdbId"].duplicated(keep=False)

    config.PROCESSED.mkdir(parents=True, exist_ok=True)
    ref.to_parquet(config.FILM_REFERENCE, index=False)
    ref.to_csv(config.FILM_REFERENCE.with_suffix(".csv"), index=False, encoding="utf-8")
    funnel(ref).to_csv(config.FILM_FUNNEL, index=False, encoding="utf-8")
    return ref


def funnel(ref: pd.DataFrame) -> pd.DataFrame:
    """Effectifs à chaque étape de la sélection, de MovieLens complet à la liste commune."""
    films = movielens.movies_with_links()
    common = ref["in_common_list"]
    cinema = common & ref["imdb_type"].eq("movie")
    steps = [
        ("Films MovieLens", len(films)),
        ("Notés au moins une fois", int((films["n_ratings"] > 0).sum())),
        (f"Notés au moins {config.MIN_RATINGS} fois", len(ref)),
        ("... avec un tmdbId", int(ref["tmdbId"].notna().sum())),
        ("... trouvés par TMDB", int(ref["tmdb_status"].eq("ok").sum())),
        ("... et retrouvés dans IMDb = liste commune", int(common.sum())),
        ("dont films de cinéma (type IMDb movie)", int(cinema.sum())),
        ("dont cinéma avec budget et recettes renseignés", int((cinema & ref["tmdb_budget_revenue"]).sum())),
    ]
    return pd.DataFrame(steps, columns=["etape", "films"])


def summary(ref: pd.DataFrame) -> str:
    flags = [c for c in ref.columns if c.startswith("flag_")]
    lines = [funnel(ref).to_string(index=False), "",
             "TMDB : " + ", ".join(f"{k} {v}" for k, v in ref["tmdb_status"].value_counts().items()),
             "IMDb : " + ", ".join(f"{k} {v}" for k, v in ref["imdb_source"].value_counts().items()),
             "Type IMDb : " + ", ".join(f"{k} {v}" for k, v in ref["imdb_type"].value_counts().items()),
             "Signalements : " + ", ".join(f"{c} {int(ref[c].sum())}" for c in flags)]
    return "\n".join(lines)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    print(summary(build()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
