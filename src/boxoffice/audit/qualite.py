"""Mesures de qualité des données : sources, entonnoir, manquants, doublons, aberrations, biais."""
import numpy as np
import pandas as pd

from boxoffice import config

SEUIL_BUDGET_CREDIBLE = 50_000
DECENNIE_FACTEUR = 3


def reference() -> pd.DataFrame:
    return pd.read_parquet(config.FILM_REFERENCE)


def films() -> pd.DataFrame:
    return pd.read_parquet(config.MOVIES_TABLE)


def sources() -> pd.DataFrame:
    liens = pd.read_csv(config.MOVIELENS_DIR / "links.csv")
    imdb = pd.read_parquet(config.IMDB_TABLE)
    caches = sum(1 for _ in config.TMDB_CACHE.glob("*.json"))
    return pd.DataFrame([
        {"source": "MovieLens ml-32m", "structure": "CSV (structuré)", "volume": f"{len(liens):,} films, 32 M notes",
         "apport": "notes des spectateurs, liens vers les deux autres sources"},
        {"source": "TMDB", "structure": "JSON (semi-structuré)", "volume": f"{caches:,} fichiers",
         "apport": "budget, recettes, genres, casting, certifications, mots-clés"},
        {"source": "IMDb", "structure": "TSV compressé (structuré)", "volume": f"{len(imdb):,} titres retenus",
         "apport": "type de titre, durée, note et votes"},
    ]).set_index("source")


def entonnoir() -> pd.DataFrame:
    etapes = pd.read_csv(config.FILM_FUNNEL)
    etapes["perdus"] = etapes["films"].shift(1) - etapes["films"]
    return etapes


def valeurs_manquantes(colonnes=("budget", "revenue", "runtime", "age_min", "collection", "main_company")) -> pd.DataFrame:
    table = films()
    return pd.DataFrame({
        "renseigné": [table[c].notna().sum() for c in colonnes],
        "manquant %": [round(100 * table[c].isna().mean(), 1) for c in colonnes],
    }, index=list(colonnes))


def zeros_deguises() -> pd.DataFrame:
    """Budget et recettes à 0 chez TMDB : des valeurs manquantes qui se font passer pour des zéros."""
    table = films()
    return pd.DataFrame([
        {"colonne": "budget", "films sans valeur exploitable": int(table["budget"].isna().sum()),
         "part du catalogue": f"{100 * table['budget'].isna().mean():.0f} %"},
        {"colonne": "revenue", "films sans valeur exploitable": int(table["revenue"].isna().sum()),
         "part du catalogue": f"{100 * table['revenue'].isna().mean():.0f} %"},
        {"colonne": "les deux à la fois", "films sans valeur exploitable":
            int((table["budget"].isna() | table["revenue"].isna()).sum()),
         "part du catalogue": f"{100 * (table['budget'].isna() | table['revenue'].isna()).mean():.0f} %"},
    ]).set_index("colonne")


def doublons() -> pd.DataFrame:
    table = reference()
    doubles = table[table["in_common_list"] & table["flag_duplicate_tmdbId"]]
    return doubles.sort_values(["tmdbId", "n_ratings"], ascending=[True, False])[
        ["movieId", "ml_title", "ml_year", "tmdbId", "tmdb_title", "n_ratings"]]


def budgets_suspects() -> pd.DataFrame:
    """Budgets incompatibles avec leur époque : comparaison au 99e centile de la décennie."""
    table = films()
    table = table[table["imdb_type"].eq("movie") & table["is_canonical"]].dropna(subset=["budget_2023"])
    table = table.assign(decennie=(table["year"] // 10 * 10).astype("Int64"))
    centile = table.groupby("decennie")["budget_2023"].transform(lambda x: x.quantile(0.99))
    mediane = table.groupby("decennie")["budget_2023"].transform("median")
    suspects = table[table["budget_2023"] > DECENNIE_FACTEUR * centile].assign(
        fois_la_mediane=(table["budget_2023"] / mediane).round(0))
    return suspects[["title", "year", "original_language", "budget", "budget_2023", "fois_la_mediane", "roi"]]


def budgets_minuscules(seuil: int = SEUIL_BUDGET_CREDIBLE) -> pd.DataFrame:
    table = films().dropna(subset=["budget", "revenue"])
    petits = table[table["budget"] < seuil]
    return petits.nsmallest(8, "budget")[["title", "year", "original_language", "budget", "revenue", "roi"]]


def annees_incoherentes() -> pd.DataFrame:
    table = reference()
    table = table[table["in_common_list"] & table["flag_annee_conflit"]] if "flag_annee_conflit" in table else \
        table[table["in_common_list"] & table["flag_year_gap_tmdb"]]
    return table[["ml_title", "ml_year", "tmdb_title", "tmdb_year", "imdb_year"]]


def certifications_brutes() -> pd.DataFrame:
    table = pd.read_parquet(config.CERTIFICATIONS_TABLE)
    comptes = table.groupby(["pays", "certification"]).size().rename("occurrences").reset_index()
    return comptes.sort_values(["pays", "occurrences"], ascending=[True, False])


def effet_inflation(annees=(1980, 1995, 2005, 2015, 2023)) -> pd.DataFrame:
    table = films().dropna(subset=["budget", "budget_2023"])
    lignes = []
    for annee in annees:
        extrait = table[table["year"] == annee]
        if len(extrait):
            lignes.append({"année": annee,
                           "budget médian (dollars de l'époque)": extrait["budget"].median(),
                           "budget médian (dollars 2023)": extrait["budget_2023"].median(),
                           "facteur": round(extrait["budget_2023"].median() / extrait["budget"].median(), 2)})
    return pd.DataFrame(lignes).set_index("année")


def biais_selection() -> pd.DataFrame:
    table = films()
    table = table[table["imdb_type"].eq("movie") & table["is_canonical"]]
    exploitables = table["budget_2023"].notna() & table["revenue_2023"].notna()
    return pd.DataFrame({
        "films exploitables": {
            "films": int(exploitables.sum()),
            "notes MovieLens (médiane)": int(table.loc[exploitables, "n_ratings"].median()),
            "année médiane": int(table.loc[exploitables, "year"].median()),
            "part en anglais": f"{100 * table.loc[exploitables, 'original_language'].eq('en').mean():.0f} %",
        },
        "films écartés": {
            "films": int((~exploitables).sum()),
            "notes MovieLens (médiane)": int(table.loc[~exploitables, "n_ratings"].median()),
            "année médiane": int(table.loc[~exploitables, "year"].median()),
            "part en anglais": f"{100 * table.loc[~exploitables, 'original_language'].eq('en').mean():.0f} %",
        },
    })


def signalements() -> pd.DataFrame:
    table = reference()
    colonnes = [c for c in table.columns if c.startswith("flag_")]
    return pd.DataFrame({"films signalés": {c.replace("flag_", ""): int(table[c].sum()) for c in colonnes}})


def notes_movielens() -> dict:
    """Volumétrie et densité des notes MovieLens : la matière première de la recommandation."""
    notes = pd.read_csv(config.MOVIELENS_RATINGS, engine="pyarrow",
                        usecols=["userId", "movieId", "rating", "timestamp"])
    par_film = notes.groupby("movieId").size()
    par_utilisateur = notes.groupby("userId").size()
    dates = pd.to_datetime(notes["timestamp"], unit="s")
    densite = 100 * len(notes) / (notes["userId"].nunique() * notes["movieId"].nunique())
    return {
        "notes": notes,
        "distribution": notes["rating"].value_counts().sort_index(),
        "par_film": par_film,
        "par_utilisateur": par_utilisateur,
        "resume": pd.DataFrame({"valeur": {
            "notes": f"{len(notes):,}".replace(",", " "),
            "utilisateurs": f"{notes['userId'].nunique():,}".replace(",", " "),
            "films notés": f"{notes['movieId'].nunique():,}".replace(",", " "),
            "période": f"{dates.min():%Y} - {dates.max():%Y}",
            "remplissage de la matrice": f"{densite:.2f} %",
            "films avec une seule note": f"{int((par_film == 1).sum()):,}".replace(",", " "),
            "notes minimum par utilisateur": int(par_utilisateur.min()),
        }}),
    }
