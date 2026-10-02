"""Figures de qualité des données."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns

VERT, ROUGE, BLEU, GRIS = "#55A868", "#C44E52", "#4C72B0", "#8C8C8C"


def entonnoir(etapes: pd.DataFrame):
    figure = px.funnel(etapes, x="films", y="etape",
                       title="De MovieLens au jeu exploitable : où partent les films")
    figure.update_layout(height=480, yaxis={"title": ""})
    return figure


def pertes(etapes: pd.DataFrame):
    table = etapes.dropna(subset=["perdus"]).copy()
    table["perdus"] = table["perdus"].astype(int)
    fig = plt.figure(figsize=(9, 4.5))
    sns.barplot(x=table["perdus"], y=table["etape"], hue=table["etape"],
                palette=[ROUGE if v > 10000 else BLEU for v in table["perdus"]], legend=False)
    plt.xlabel("films perdus à cette étape")
    plt.ylabel("")
    plt.title("Deux étapes concentrent presque toutes les pertes")
    plt.tight_layout()
    return fig


def valeurs_manquantes(table: pd.DataFrame):
    fig = plt.figure(figsize=(8, 4))
    ordre = table.sort_values("manquant %", ascending=False)
    sns.barplot(x=ordre["manquant %"], y=ordre.index, hue=ordre.index,
                palette=[ROUGE if v > 50 else BLEU for v in ordre["manquant %"]], legend=False)
    plt.xlabel("% de films sans valeur exploitable")
    plt.ylabel("")
    plt.title("Ce que TMDB ne renseigne pas")
    for position, valeur in enumerate(ordre["manquant %"]):
        plt.text(valeur + 1, position, f"{valeur:.0f} %", va="center")
    plt.tight_layout()
    return fig


def distribution_budget(films: pd.DataFrame, seuil_credible: int = 50_000):
    valeurs = films["budget"].dropna()
    fig = plt.figure(figsize=(9, 4))
    plt.hist(valeurs, bins=np.logspace(2, 9, 60), color=BLEU)
    plt.xscale("log")
    plt.axvline(seuil_credible, color=ROUGE, linestyle="--",
                label=f"seuil de crédibilité : {seuil_credible:,} $".replace(",", " "))
    plt.xlabel("budget déclaré (échelle logarithmique)")
    plt.ylabel("films")
    plt.title("Des budgets déclarés à quelques milliers de dollars")
    plt.legend()
    plt.tight_layout()
    return fig


def inflation(films: pd.DataFrame):
    table = films.dropna(subset=["budget", "budget_2023"])
    table = table[table["year"].between(1980, 2023)]
    medianes = table.groupby("year")[["budget", "budget_2023"]].median() / 1e6
    fig = plt.figure(figsize=(9, 4.5))
    plt.plot(medianes.index, medianes["budget"], marker="o", label="dollars de l'époque", color=GRIS)
    plt.plot(medianes.index, medianes["budget_2023"], marker="o", label="dollars constants 2023", color=BLEU)
    plt.xlabel("année de sortie")
    plt.ylabel("budget médian (millions de dollars)")
    plt.title("Sans correction de l'inflation, les décennies ne sont pas comparables")
    plt.legend()
    plt.tight_layout()
    return fig


def biais_selection(films: pd.DataFrame):
    table = films[films["imdb_type"].eq("movie") & films["is_canonical"]].copy()
    table["groupe"] = np.where(table["budget_2023"].notna() & table["revenue_2023"].notna(),
                               "films exploitables", "films écartés")
    fig = plt.figure(figsize=(9, 4.5))
    sns.boxplot(data=table, x="n_ratings", y="groupe", hue="groupe",
                palette={"films exploitables": VERT, "films écartés": ROUGE}, legend=False)
    plt.xscale("log")
    plt.xlabel("nombre de notes MovieLens (échelle logarithmique)")
    plt.ylabel("")
    plt.title("Biais de sélection : les films exploitables sont les plus vus")
    plt.tight_layout()
    return fig


def sources_par_annee(reference: pd.DataFrame):
    table = reference[reference["in_common_list"]].dropna(subset=["ml_year", "tmdb_year"]).copy()
    table["ecart"] = (table["tmdb_year"] - table["ml_year"]).astype(float)
    ecarts = table["ecart"].clip(-10, 10)
    fig = plt.figure(figsize=(8, 4))
    plt.hist(ecarts, bins=41, color=BLEU)
    plt.yscale("log")
    plt.xlabel("écart entre l'année TMDB et l'année MovieLens")
    plt.ylabel("films (échelle logarithmique)")
    plt.title("Les trois sources ne datent pas toujours les films de la même façon")
    plt.tight_layout()
    return fig


def notes_et_longue_traine(mesures: dict):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    distribution = mesures["distribution"] / mesures["distribution"].sum() * 100
    axes[0].bar([str(note) for note in distribution.index], distribution.values, color=BLEU)
    axes[0].set(xlabel="note attribuée", ylabel="% des notes",
                title="Les spectateurs notent rarement en dessous de 2")

    axes[1].hist(mesures["par_film"], bins=np.logspace(0, 5.5, 60), color=BLEU)
    axes[1].set(xscale="log", yscale="log", xlabel="notes reçues par film (échelle log)",
                ylabel="films (échelle log)", title="Une longue traîne : peu de films très notés")
    axes[1].axvline(10, color=ROUGE, linestyle="--", label="seuil retenu : 10 notes")
    axes[1].legend()
    plt.tight_layout()
    return fig
