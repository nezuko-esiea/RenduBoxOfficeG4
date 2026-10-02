"""Téléchargement idempotent de MovieLens et sélection des films du projet."""
import logging
import os
import zipfile
from pathlib import Path

import pandas as pd
import requests

from boxoffice import config

log = logging.getLogger(__name__)


def dataset_dir(variant: str = config.MOVIELENS_VARIANT) -> Path:
    return config.RAW / "movielens" / variant


def download(variant: str = config.MOVIELENS_VARIANT, force: bool = False) -> Path:
    """Télécharge et extrait une variante, sauf si elle est déjà présente."""
    target = dataset_dir(variant)
    if (target / "links.csv").exists() and not force:
        return target
    url = f"https://files.grouplens.org/datasets/movielens/{variant}.zip"
    archive_path = target.parent / f"{variant}.zip.part"
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    log.info("Téléchargement de %s", url)
    with requests.get(url, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        with archive_path.open("wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(target.parent)
    os.remove(archive_path)
    return target


def rating_counts(variant: str = config.MOVIELENS_VARIANT) -> pd.Series:
    """Nombre de notes par movieId, calculé une fois puis relu depuis data/interim."""
    cache = config.INTERIM / f"{variant}_rating_counts.parquet"
    if cache.exists():
        return pd.read_parquet(cache)["n_ratings"]
    ratings = pd.read_csv(download(variant) / "ratings.csv", engine="pyarrow", usecols=["movieId"])
    counts = ratings.groupby("movieId").size().rename("n_ratings")
    cache.parent.mkdir(parents=True, exist_ok=True)
    counts.to_frame().to_parquet(cache)
    return counts


def movies_with_links(variant: str = config.MOVIELENS_VARIANT) -> pd.DataFrame:
    """Tous les films de la variante avec identifiants et nombre de notes (0 si jamais noté)."""
    folder = download(variant)
    movies = pd.read_csv(folder / "movies.csv")
    links = pd.read_csv(folder / "links.csv", dtype={"imdbId": str, "tmdbId": "Int64"})
    films = movies.merge(links, on="movieId", how="left").merge(rating_counts(variant), on="movieId", how="left")
    films["n_ratings"] = films["n_ratings"].fillna(0).astype(int)
    return films


def selected_movies(variant: str = config.MOVIELENS_VARIANT, min_ratings: int = config.MIN_RATINGS) -> pd.DataFrame:
    """Films retenus pour le projet : notés au moins `min_ratings` fois."""
    films = movies_with_links(variant)
    return films[films["n_ratings"] >= min_ratings].reset_index(drop=True)


def tmdb_ids(variant: str = config.MOVIELENS_VARIANT) -> list[int]:
    """Identifiants TMDB distincts des films retenus (les films sans tmdbId sont ignorés)."""
    return sorted(selected_movies(variant)["tmdbId"].dropna().astype(int).unique().tolist())


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    films = selected_movies()
    log.info("%s : %d films notés au moins %d fois", config.MOVIELENS_VARIANT, len(films), config.MIN_RATINGS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
