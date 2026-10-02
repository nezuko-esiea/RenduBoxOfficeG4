"""Tables propres issues du cache TMDB : films, genres, personnes, certifications, mots-clés."""
import argparse
import json
import logging
from collections import Counter

import pandas as pd

from boxoffice import config
from boxoffice.ingestion import cpi, tmdb

log = logging.getLogger(__name__)

MIN_BUDGET = 1_000
MIN_REVENUE = 1_000
MIN_RUNTIME = 1
CAST_TOP = 3
HIT_ROI = 2.5
COVID_YEARS = (2020, 2021)
CERTIFICATION_COUNTRIES = ("FR", "US")
FR_AGE = {"TP": 0, "U": 0, "10": 10, "12": 12, "16": 16, "18": 18}
US_AGE = {"G": 0, "PG": 0, "PG-13": 13, "R": 17, "NC-17": 18}


def _positive(value, minimum):
    return value if value and value >= minimum else None


def _entier(value):
    return None if value is None or pd.isna(value) else int(value)


def _annee(film: dict, movie: dict) -> tuple:
    date = movie.get("release_date") or ""
    sources = {"ml": _entier(film.get("ml_year")),
               "tmdb": _entier(date[:4]) if date else None,
               "imdb": _entier(film.get("imdb_year"))}
    valeurs = [annee for annee in sources.values() if annee is not None]
    if not valeurs:
        return None, False
    compte = Counter(valeurs)
    majoritaire, votes = compte.most_common(1)[0]
    if votes > 1:
        return majoritaire, len(compte) > 1
    return sources["imdb"] or sources["tmdb"] or sources["ml"], True


def _certification_rows(movie: dict) -> list[dict]:
    rows = []
    for country in movie.get("release_dates", {}).get("results", []):
        if country["iso_3166_1"] not in CERTIFICATION_COUNTRIES:
            continue
        for release in country["release_dates"]:
            code = (release.get("certification") or "").strip()
            if code and code != "NR":
                rows.append({"pays": country["iso_3166_1"], "certification": code,
                             "type_sortie": release.get("type"),
                             "date_sortie": (release.get("release_date") or "")[:10]})
    return rows


def _ages(certifications: list[dict]) -> dict:
    ages = {}
    for pays, table in (("FR", FR_AGE), ("US", US_AGE)):
        connus = [table[c["certification"]] for c in certifications
                  if c["pays"] == pays and c["certification"] in table]
        ages[pays] = max(connus) if connus else None
    connus = [age for age in ages.values() if age is not None]
    ages["min"] = max(connus) if connus else None
    return ages


def _movie_row(movie: dict, film: dict, factors: pd.Series) -> dict:
    date = movie.get("release_date") or ""
    year, conflit_annee = _annee(film, movie)
    budget = _positive(movie.get("budget"), MIN_BUDGET)
    revenue = _positive(movie.get("revenue"), MIN_REVENUE)
    factor = factors.get(year) if year else None
    companies = movie.get("production_companies") or []
    collection = movie.get("belongs_to_collection") or {}
    ages = _ages(_certification_rows(movie))
    return {
        "movieId": film["movieId"],
        "tmdbId": film["tmdbId"],
        "imdb_id": film["imdb_id"],
        "title": movie.get("title"),
        "year": year,
        "year_tmdb": int(date[:4]) if date else None,
        "flag_annee_conflit": conflit_annee,
        "month": int(date[5:7]) if date else None,
        "release_date": date or None,
        "periode_covid": year in COVID_YEARS if year else False,
        "runtime": _positive(movie.get("runtime"), MIN_RUNTIME),
        "budget": budget,
        "revenue": revenue,
        "budget_2023": round(budget * factor) if budget and factor else None,
        "revenue_2023": round(revenue * factor) if revenue and factor else None,
        "roi": revenue / budget if budget and revenue else None,
        "is_hit": revenue / budget >= HIT_ROI if budget and revenue else None,
        "is_franchise": bool(collection),
        "collection": collection.get("name"),
        "original_language": movie.get("original_language"),
        "origin_country": (movie.get("origin_country") or [None])[0],
        "n_companies": len(companies),
        "main_company": companies[0]["name"] if companies else None,
        "n_cast": len(movie.get("credits", {}).get("cast", [])),
        "age_min": ages["min"],
        "age_min_fr": ages["FR"],
        "age_min_us": ages["US"],
        "vote_average": movie.get("vote_average"),
        "vote_count": movie.get("vote_count"),
        "popularity": movie.get("popularity"),
        "imdb_type": film["imdb_type"],
        "n_ratings": film["n_ratings"],
    }


def build() -> dict[str, pd.DataFrame]:
    reference = pd.read_parquet(config.FILM_REFERENCE)
    films = reference[reference["in_common_list"]]
    factors = cpi.factors()
    movies, genres, people, certifications, keywords = [], [], [], [], []

    for n, film in enumerate(films.to_dict("records"), 1):
        movie = json.loads(tmdb.cache_path(config.TMDB_CACHE, int(film["tmdbId"])).read_text(encoding="utf-8"))
        movie_id = film["movieId"]
        movies.append(_movie_row(movie, film, factors))
        genres += [{"movieId": movie_id, "genre_id": g["id"], "genre": g["name"]}
                   for g in movie.get("genres") or []]
        keywords += [{"movieId": movie_id, "keyword_id": k["id"], "keyword": k["name"]}
                     for k in movie.get("keywords", {}).get("keywords") or []]
        certifications += [{"movieId": movie_id, **row} for row in _certification_rows(movie)]
        credits = movie.get("credits", {})
        people += [{"movieId": movie_id, "person_id": c["id"], "nom": c["name"], "role": "acteur",
                    "ordre": c.get("order"), "personnage": c.get("character")}
                   for c in credits.get("cast", []) if (c.get("order") or 0) < CAST_TOP]
        people += [{"movieId": movie_id, "person_id": c["id"], "nom": c["name"], "role": "realisateur",
                    "ordre": None, "personnage": None}
                   for c in credits.get("crew", []) if c.get("job") == "Director"]
        if n % 5000 == 0:
            log.info("%d/%d films", n, len(films))

    table_movies = pd.DataFrame(movies)
    canoniques = set(table_movies.sort_values("n_ratings", ascending=False).drop_duplicates("tmdbId")["movieId"])
    table_movies["is_canonical"] = table_movies["movieId"].isin(canoniques)

    tables = {
        config.MOVIES_TABLE: table_movies.astype({"year": "Int64", "year_tmdb": "Int64",
                                                          "month": "Int64", "runtime": "Int64",
                                                          "budget": "Int64", "revenue": "Int64",
                                                          "budget_2023": "Int64", "revenue_2023": "Int64",
                                                          "age_min": "Int64", "age_min_fr": "Int64",
                                                          "age_min_us": "Int64", "is_hit": "boolean"}),
        config.GENRES_TABLE: pd.DataFrame(genres),
        config.PEOPLE_TABLE: pd.DataFrame(people).astype({"ordre": "Int64"}),
        config.CERTIFICATIONS_TABLE: pd.DataFrame(certifications).astype({"type_sortie": "Int64"}),
        config.KEYWORDS_TABLE: pd.DataFrame(keywords),
    }
    config.PROCESSED.mkdir(parents=True, exist_ok=True)
    for path, table in tables.items():
        table.to_parquet(path, index=False)
        table.to_csv(path.with_suffix(".csv"), index=False, encoding="utf-8")
    return {path.stem: table for path, table in tables.items()}


def summary(tables: dict[str, pd.DataFrame]) -> str:
    movies = tables["movies"]
    money = movies.dropna(subset=["budget", "revenue"])
    lines = [" | ".join(f"{name} {len(table):,}" for name, table in tables.items()), ""]
    lines.append(f"budget renseigné : {movies['budget'].notna().sum():,} | "
                 f"recettes : {movies['revenue'].notna().sum():,} | les deux : {len(money):,}")
    lines.append(f"films rentables (ROI >= {HIT_ROI}) : {money['is_hit'].sum():,} "
                 f"({100 * money['is_hit'].mean():.1f} %) | ROI médian : {money['roi'].median():.2f}")
    lines.append(f"âge minimum connu : {movies['age_min'].notna().sum():,} "
                 f"({100 * movies['age_min'].notna().mean():.1f} %) | "
                 f"France {movies['age_min_fr'].notna().sum():,} | "
                 f"États-Unis {movies['age_min_us'].notna().sum():,}")
    lines.append("âge minimum : " + ", ".join(f"{int(k)} ans : {v}" for k, v in movies["age_min"].value_counts().sort_index().items()))
    lines.append(f"franchises : {movies['is_franchise'].sum():,} | période COVID : {movies['periode_covid'].sum():,}")
    lines.append(f"années en conflit entre sources : {movies['flag_annee_conflit'].sum():,} | "
                 f"corrigées par rapport à TMDB : {(movies['year'] != movies['year_tmdb']).sum():,} | "
                 f"films non canoniques (doublons) : {(~movies['is_canonical']).sum():,}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description="Construit les tables propres du projet.").parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    print(summary(build()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
