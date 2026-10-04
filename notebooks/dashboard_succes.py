"""Tableau de bord métier : estimer la rentabilité d'un projet de film avant production."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from boxoffice.features import success
from boxoffice.models import entrainement, succes as persistance
from boxoffice.viz import metier

VERT, ORANGE, ROUGE = "#55A868", "#DD8452", "#C44E52"
SAISONS = {"Été (juin-août)": "ete", "Fêtes de fin d'année": "fin_annee", "Printemps": "printemps",
           "Rentrée (septembre)": "rentree", "Automne": "automne", "Hiver": "hiver"}
AGES = {"Tous publics": 0, "Déconseillé aux moins de 13 ans": 13, "Interdit aux moins de 17 ans": 17}

st.set_page_config(page_title="Quel film produire ?", page_icon="🎬", layout="wide")


@st.cache_resource(show_spinner="Chargement du modèle et des données...")
def charger():
    data = success.charger()
    variables = success.colonnes_variables(data)
    jeu = entrainement.preparer(data, variables)
    modele, carte = persistance.charger()
    probabilites = modele.predict_proba(jeu["X"].loc[jeu["i_test"]])[:, 1]
    return data, variables, jeu, modele, carte, probabilites


def zone(probabilite: float) -> tuple[str, str]:
    if probabilite >= entrainement.SEUIL_RECOMMANDATION:
        return "RECOMMANDER", VERT
    if probabilite < entrainement.SEUIL_REJET:
        return "ÉCARTER", ROUGE
    return "EXAMINER", ORANGE


def jauge(probabilite: float, couleur: str) -> go.Figure:
    figure = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(100 * probabilite),
        number={"suffix": " %", "font": {"size": 44}},
        title={"text": "Chances d'être rentable"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": couleur},
            "steps": [{"range": [0, 30], "color": "#F7DDDC"},
                      {"range": [30, 70], "color": "#FCEBDC"},
                      {"range": [70, 100], "color": "#DDEEE2"}],
            "threshold": {"line": {"color": "black", "width": 3}, "value": 35},
        }))
    figure.update_layout(height=320, margin={"t": 60, "b": 10})
    return figure


data, variables, jeu, modele, carte, probabilites = charger()
genres = sorted(c[6:] for c in variables if c.startswith("genre_") and data[c].sum() >= 300)
taux_de_base = jeu["y"][jeu["i_test"]].mean()

st.title("Quel film produire ensuite ?")
st.caption("Outil d'aide à la décision fondé sur 9 268 films de cinéma sortis entre 1915 et 2023, "
           "analysés à partir de MovieLens, TMDB et IMDb.")

colonnes = st.columns(4)
colonnes[0].metric("Films analysés", "9 268")
colonnes[1].metric("Rentables dans le catalogue", f"{100 * taux_de_base:.0f} %")
colonnes[2].metric("Rentables parmi les projets recommandés", "72 %", "+37 points")
colonnes[3].metric("Seuil de rentabilité", "2,5 fois le budget")

st.divider()

st.header("Simulateur de projet")
st.write("Décrivez un projet de film : l'outil estime sa probabilité de rentabilité et propose une décision.")

gauche, droite = st.columns([1, 1.4])

with gauche:
    genre = st.selectbox("Genre principal", genres, index=genres.index("Action") if "Action" in genres else 0)
    budget = st.slider("Budget de production (millions de dollars)", 2, 300, 60, step=2)
    saga = st.toggle("Suite ou univers existant", value=False)
    saison = st.selectbox("Période de sortie", list(SAISONS))
    duree = st.slider("Durée (minutes)", 70, 200, 110, step=5)
    classification = st.selectbox("Classification d'âge visée", list(AGES), index=1)
    studio = st.radio("Production", ["Grand studio", "Indépendant"], horizontal=True)

specification = {
    "nom": "projet", "genre": genre, "budget_millions": budget, "franchise": int(saga),
    "saison": SAISONS[saison], "studio": "major" if studio == "Grand studio" else "autre",
    "age": AGES[classification], "duree": duree,
}
ligne = entrainement.projet(data, jeu, **specification)
probabilite = modele.predict_proba(pd.DataFrame([ligne]).astype("float64")[variables])[0, 1]
decision, couleur = zone(probabilite)

with droite:
    st.plotly_chart(jauge(probabilite, couleur), width="stretch")
    st.markdown(f"<h2 style='text-align:center;color:{couleur}'>{decision}</h2>", unsafe_allow_html=True)
    if decision == "RECOMMANDER":
        st.success("Ce projet cumule les caractéristiques gagnantes. Parmi les projets ainsi classés, "
                   "72 % se révèlent rentables.")
    elif decision == "EXAMINER":
        st.warning("Signaux contradictoires : l'outil ne tranche pas. Dans cette zone, 40 % des films "
                   "sont rentables, soit presque un pile ou face. Une analyse humaine est nécessaire.")
    else:
        st.error("Configuration historiquement perdante : seuls 15 % des projets de cette zone "
                 "se révèlent rentables.")

st.divider()

st.header("Ce qui fait la différence")
onglets = st.tabs(["Budget et saga", "Genres et budgets", "Calendrier", "Fiabilité de l'outil",
                   "Erreurs par type de film"])

with onglets[0]:
    courbes = entrainement.courbe_budget_saga(modele, data, jeu, genre=genre)
    st.plotly_chart(metier.courbe_budget_saga(courbes), width="stretch")
    st.info("Pour un film isolé, la probabilité de rentabilité **diminue** quand le budget augmente. "
            "Pour une suite, elle se maintient. Un budget élevé ne se justifie que sur une licence existante.")

with onglets[1]:
    grille = entrainement.grille_genre_budget(modele, data, jeu, genres)
    st.plotly_chart(metier.carte_genre_budget(grille.sort_values("30 M$", ascending=False),
                                              titre="Chances de rentabilité, film isolé, sortie estivale"),
                    width="stretch")
    st.info("La colonne de gauche est verte, celle de droite est rouge : quel que soit le genre, "
            "le petit budget est plus sûr.")

with onglets[2]:
    st.plotly_chart(metier.saisonnalite(data), width="stretch")
    st.info("Juin et juillet dominent, suivis des fêtes de fin d'année. Septembre est le pire mois, "
            "avec près de vingt points d'écart avec l'été.")

with onglets[3]:
    zones = entrainement.zones(jeu, probabilites)
    st.plotly_chart(metier.zones_decision(zones, taux_de_base), width="stretch")
    st.info("Testé sur 983 films récents que l'outil n'avait jamais vus. Il double les chances de succès "
            "sur la zone haute, et surtout il écarte 360 projets dont seuls 15 % étaient rentables.")

with onglets[4]:
    fiable = entrainement.fiabilite_par_genre(data, jeu, probabilites)
    st.plotly_chart(metier.fiabilite_par_genre(fiable), width="stretch")
    meilleur, pire = fiable.index[0], fiable.index[-1]
    biaise = fiable["biais"].abs().idxmax()
    sens = "surestime" if fiable.loc[biaise, "biais"] > 0 else "sous-estime"
    st.info(
        f"Le **{meilleur}** est le mieux prédit (Brier {fiable.loc[meilleur, 'Brier']:.2f}) : ses deux barres "
        f"se superposent presque, donc ce que l'outil annonce correspond au taux réel. À l'inverse, le "
        f"**{pire}** est le moins fiable (Brier {fiable.loc[pire, 'Brier']:.2f}). Le biais le plus marqué "
        f"touche le **{biaise}**, que l'outil {sens} de {abs(fiable.loc[biaise, 'biais']):.0%}.")
    st.plotly_chart(metier.residus_par_genre(entrainement.residus_par_genre(data, jeu, probabilites)),
                    width="stretch")
    st.info("Un résidu est l'écart entre la probabilité annoncée et l'issue réelle. Une boîte centrée "
            "au-dessus de zéro signale une surestimation systématique ; plus la boîte est large, plus "
            "les prédictions sont dispersées.")

st.divider()

with st.expander("Ce que cet outil ne fait pas"):
    st.markdown("""
- **Il donne une probabilité, pas une garantie** : environ trois recommandations sur dix se trompent.
- **Il ignore ce qu'il ne mesure pas** : qualité du scénario, budget marketing, concurrence à la date de sortie.
- **Il décrit le cinéma financé et distribué à l'international**, seul périmètre où budgets et recettes sont publics.
- **Il n'anticipe pas les ruptures de marché** : pendant la fermeture des salles en 2020 et 2021,
  sa fiabilité est tombée de moitié.

**Usage recommandé** : classer les projets en amont et écarter les configurations les plus risquées.
Il éclaire la décision, il ne la remplace pas.
""")

st.caption(f"Modèle : {carte['modele']} · entraîné le {carte['entraine_le'][:10]} · "
           f"AUC {carte['metriques']['AUC']} sur un test temporel (films sortis à partir de 2018)")
