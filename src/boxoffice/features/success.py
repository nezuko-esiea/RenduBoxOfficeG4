"""Tableau d'entraînement du modèle de succès : cibles is_hit et log_roi, variables pré-production."""
import argparse
import logging

import numpy as np
import pandas as pd

from boxoffice import config

log = logging.getLogger(__name__)

HIT_ROI = 2.5
PART_STUDIO = 0.5
PART_MARKETING = 0.25
TOP_CAST = 3
TOP_KEYWORDS = 50
MAJORS = ("Paramount", "Universal", "Columbia", "Warner Bros", "20th Century", "Walt Disney",
          "Metro-Goldwyn-Mayer", "New Line", "Sony Pictures", "Touchstone", "TriStar", "Lionsgate",
          "Miramax", "DreamWorks", "Amblin", "Orion", "Fox Searchlight", "Focus Features")
SAISONS = {1: "hiver", 2: "hiver", 3: "printemps", 4: "printemps", 5: "printemps", 6: "ete",
           7: "ete", 8: "ete", 9: "rentree", 10: "automne", 11: "fin_annee", 12: "fin_annee"}
FUITE = ("revenue", "revenue_2023", "roi", "vote_average", "vote_count", "popularity", "n_ratings")


def point_mort(part_studio: float = PART_STUDIO, marketing: float = PART_MARKETING) -> float:
    """Multiple du budget à atteindre en recettes pour que le studio couvre production et marketing."""
    return (1 + marketing) / part_studio


def grille_point_mort(parts_studio=(0.50, 0.45, 0.40), marketings=(0.0, 0.25, 0.50)) -> pd.DataFrame:
    grille = pd.DataFrame({f"marketing = {100 * m:.0f} % du budget": [point_mort(p, m) for p in parts_studio]
                           for m in marketings},
                          index=[f"le studio touche {100 * p:.0f} % des recettes" for p in parts_studio])
    return grille.round(2)


def resultat_studio(multiples=(1, 1.5, 2, 2.5, 3, 4), budget: float = 100,
                    part_studio: float = PART_STUDIO, marketing: float = PART_MARKETING) -> pd.DataFrame:
    """Résultat du studio, en millions, pour un film de budget donné selon le ROI atteint en salles."""
    lignes = [{"ROI": multiple, "recettes en salles": multiple * budget,
               "part du studio": part_studio * multiple * budget,
               "production": budget, "marketing": marketing * budget,
               "résultat du studio": part_studio * multiple * budget - budget - marketing * budget}
              for multiple in multiples]
    return pd.DataFrame(lignes).set_index("ROI")


def films_du_modele() -> pd.DataFrame:
    movies = pd.read_parquet(config.MOVIES_TABLE)
    films = movies[movies["imdb_type"].eq("movie") & movies["is_canonical"]]
    return films[films["release_date"].notna()].copy()


def _notoriete(films: pd.DataFrame) -> pd.DataFrame:
    people = pd.read_parquet(config.PEOPLE_TABLE)
    people = people[(people["role"].eq("realisateur")) | (people["ordre"] < TOP_CAST)]
    historique = people.merge(films[["movieId", "release_date", "is_hit"]], on="movieId")
    historique = historique.sort_values(["release_date", "movieId"])

    par_personne = historique.groupby("person_id")
    historique["films_avant"] = par_personne.cumcount()
    connu = historique["is_hit"].fillna(False).astype(int)
    historique["hits_avant"] = par_personne["is_hit"].transform(lambda s: s.fillna(False).astype(int).cumsum()) - connu
    historique["notes_avant"] = par_personne["is_hit"].transform(lambda s: s.notna().astype(int).cumsum()) - historique["is_hit"].notna().astype(int)
    historique["taux_hits_avant"] = np.where(historique["notes_avant"] > 0,
                                             historique["hits_avant"] / historique["notes_avant"].replace(0, np.nan), np.nan)

    colonnes = ["films_avant", "taux_hits_avant"]
    realisateurs = historique[historique["role"].eq("realisateur")].groupby("movieId")[colonnes].max()
    acteurs = historique[historique["role"].eq("acteur")].groupby("movieId")[colonnes].mean()
    return realisateurs.add_prefix("director_").join(acteurs.add_prefix("cast_"), how="outer")


def _genres(films: pd.DataFrame) -> pd.DataFrame:
    genres = pd.read_parquet(config.GENRES_TABLE)
    genres = genres[genres["movieId"].isin(films["movieId"])]
    table = pd.crosstab(genres["movieId"], genres["genre"]).astype("int8")
    return table.add_prefix("genre_")


def _keywords(films: pd.DataFrame) -> pd.DataFrame:
    keywords = pd.read_parquet(config.KEYWORDS_TABLE)
    keywords = keywords[keywords["movieId"].isin(films["movieId"])]
    frequents = keywords["keyword"].value_counts().head(TOP_KEYWORDS).index
    table = pd.crosstab(keywords.loc[keywords["keyword"].isin(frequents), "movieId"],
                        keywords.loc[keywords["keyword"].isin(frequents), "keyword"]).astype("int8")
    return table.add_prefix("kw_")


def build(avec_keywords: bool = False) -> pd.DataFrame:
    films = films_du_modele()
    notoriete = _notoriete(films)

    entraine = films.dropna(subset=["budget_2023", "revenue_2023"]).copy()
    entraine = entraine[entraine["roi"] > 0]
    dataset = pd.DataFrame({
        "movieId": entraine["movieId"],
        "title": entraine["title"],
        "year": entraine["year"],
        "is_hit": entraine["roi"] >= HIT_ROI,
        "log_roi": np.log(entraine["roi"].astype(float)),
        "log_budget": np.log(entraine["budget_2023"].astype(float)),
        "runtime": entraine["runtime"].astype("Float64"),
        "saison": entraine["month"].map(SAISONS),
        "month": entraine["month"].astype("Int64"),
        "is_franchise": entraine["is_franchise"].astype(int),
        "langue": np.where(entraine["original_language"].eq("en"), "en", "autre"),
        "pays": entraine["origin_country"].fillna("inconnu"),
        "studio": np.where(entraine["main_company"].fillna("").str.startswith(MAJORS), "major", "autre"),
        "n_cast": entraine["n_cast"],
        "age_min": entraine["age_min"].astype("Float64"),
        "periode_covid": entraine["periode_covid"].astype(int),
    }).set_index("movieId")

    dataset = dataset.join(notoriete).join(_genres(entraine))
    if avec_keywords:
        dataset = dataset.join(_keywords(entraine))
    dataset[notoriete.columns] = dataset[notoriete.columns].fillna({c: 0 for c in notoriete.columns if c.endswith("films_avant")})
    pays_frequents = dataset["pays"].value_counts().head(10).index
    dataset["pays"] = dataset["pays"].where(dataset["pays"].isin(pays_frequents), "autre")
    return dataset.reset_index()


def charger(recalculer: bool = False, avec_keywords: bool = False) -> pd.DataFrame:
    """Tableau d'entraînement, relu depuis data/processed ou reconstruit puis enregistré."""
    if config.SUCCESS_DATASET.exists() and not recalculer:
        return pd.read_parquet(config.SUCCESS_DATASET)
    dataset = build(avec_keywords=avec_keywords)
    config.PROCESSED.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(config.SUCCESS_DATASET, index=False)
    dataset.to_csv(config.SUCCESS_DATASET.with_suffix(".csv"), index=False, encoding="utf-8")
    return dataset


def colonnes_variables(dataset: pd.DataFrame) -> list[str]:
    exclues = {"movieId", "title", "year", "is_hit", "log_roi"}
    return [c for c in dataset.columns if c not in exclues]


def decoupage_temporel(dataset: pd.DataFrame, annee_test: int = 2018) -> tuple:
    train = dataset[dataset["year"] < annee_test]
    test = dataset[dataset["year"] >= annee_test]
    return train, test


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Construit le tableau d'entraînement du modèle de succès.")
    parser.add_argument("--keywords", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    dataset = charger(recalculer=True, avec_keywords=args.keywords)
    train, test = decoupage_temporel(dataset)
    variables = colonnes_variables(dataset)
    print(f"{len(dataset):,} films | {len(variables)} variables | {100 * dataset['is_hit'].mean():.1f} % de succès")
    print(f"entraînement {len(train):,} films (< 2018) | test {len(test):,} films (>= 2018)")
    print("variables :", ", ".join(variables[:12]), "...")
    manquants = dataset[variables].isna().mean().sort_values(ascending=False)
    print("valeurs manquantes :", ", ".join(f"{c} {100 * v:.1f} %" for c, v in manquants.head(5).items() if v > 0) or "aucune")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
