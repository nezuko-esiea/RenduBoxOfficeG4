"""Figures destinées à l'audience métier (Plotly, interactives)."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio

COULEURS = {"non rentable": "#C44E52", "rentable": "#55A868"}
ZONES = {"écarter": "#C44E52", "examiner": "#DD8452", "recommander": "#55A868"}


def activer_affichage() -> None:
    """Intègre la librairie Plotly au notebook pour que les figures survivent à un export HTML."""
    pio.renderers.default = "notebook"


def nuage_films(data: pd.DataFrame, seuil_roi: float = 2.5):
    nuage = data.assign(budget=np.exp(data["log_budget"]), roi=np.exp(data["log_roi"]),
                        rentabilite=np.where(data["is_hit"], "rentable", "non rentable"))
    figure = px.scatter(
        nuage, x="budget", y="roi", color="rentabilite", color_discrete_map=COULEURS,
        hover_name="title", hover_data={"year": True, "budget": ":,.0f", "roi": ":.2f", "rentabilite": False},
        log_x=True, log_y=True, opacity=0.45,
        labels={"budget": "budget (dollars constants 2023)", "roi": "ROI", "rentabilite": ""},
        title="Chaque point est un film : budget contre rentabilité")
    figure.add_hline(y=seuil_roi, line_dash="dash", line_color="grey", annotation_text="seuil de rentabilité")
    figure.update_layout(height=520)
    return figure


def carte_genre_budget(matrice: pd.DataFrame, titre: str = "Probabilité de rentabilité estimée par le modèle"):
    figure = px.imshow(matrice, text_auto=True, aspect="auto", color_continuous_scale="RdYlGn",
                       labels={"x": "budget", "y": "", "color": "% de chances"}, title=titre)
    figure.update_layout(height=520)
    return figure


def nuage_erreurs(diagnostic: pd.DataFrame, seuil: float):
    table = diagnostic.assign(realite=np.where(diagnostic["rentable"] == 1, "rentable", "non rentable"))
    figure = px.strip(table, x="probabilite", y="realite", color="realite", color_discrete_map=COULEURS,
                      hover_name="film", hover_data={"annee": True, "budget_millions": ":.0f", "realite": False},
                      stripmode="overlay", labels={"probabilite": "probabilité prédite", "realite": ""},
                      title="Où le modèle se trompe : chaque point est un film du test")
    figure.add_vline(x=seuil, line_dash="dash", line_color="grey", annotation_text="seuil de décision")
    figure.update_traces(jitter=0.8, marker={"size": 7, "opacity": 0.6})
    figure.update_layout(height=420, showlegend=False)
    return figure


def zones_decision(resume: pd.DataFrame, taux_de_base: float):
    table = resume.reset_index()
    table["taux"] = (100 * table["taux_reel"]).round(0)
    figure = px.bar(table, x="zone", y="taux", text="taux", color="zone", color_discrete_map=ZONES,
                    labels={"taux": "% de films rentables", "zone": ""}, hover_data={"films": True},
                    title="Ce que vaut chaque zone de décision")
    figure.add_hline(y=100 * taux_de_base, line_dash="dash",
                     annotation_text=f"taux de base : {100 * taux_de_base:.0f} %")
    figure.update_traces(texttemplate="%{text} %", textposition="outside")
    figure.update_layout(height=450, showlegend=False)
    return figure


def courbe_budget_saga(courbes: pd.DataFrame, seuil_pourcent: float = 70):
    figure = px.line(courbes, x="budget", y="probabilité", color="type de film", markers=True,
                     color_discrete_map={"film isolé": "#C44E52", "suite ou univers existant": "#55A868"},
                     labels={"budget": "budget (millions de dollars 2023)",
                             "probabilité": "% de chances d'être rentable"},
                     title="Un gros budget ne se justifie que sur une franchise")
    figure.add_hline(y=seuil_pourcent, line_dash="dash", annotation_text="seuil de recommandation")
    figure.update_layout(height=470)
    return figure


def projets(estimation: pd.DataFrame):
    table = estimation.reset_index(names="projet")
    table["pourcentage"] = (100 * table["probabilité de rentabilité"]).round(0)
    figure = px.bar(table, x="pourcentage", y="projet", orientation="h", color="décision",
                    color_discrete_map=ZONES, text="pourcentage",
                    labels={"pourcentage": "% de chances d'être rentable", "projet": ""},
                    title="Six projets soumis au modèle")
    figure.update_traces(texttemplate="%{text} %", textposition="outside")
    figure.update_layout(height=450, yaxis={"categoryorder": "total ascending"})
    return figure


def entonnoir(etapes: pd.DataFrame):
    figure = px.funnel(etapes, x="films", y="etape", title="De MovieLens au jeu exploitable")
    figure.update_layout(height=460)
    return figure


def saisonnalite(data: pd.DataFrame):
    saisons = {1: "janvier", 2: "février", 3: "mars", 4: "avril", 5: "mai", 6: "juin",
               7: "juillet", 8: "août", 9: "septembre", 10: "octobre", 11: "novembre", 12: "décembre"}
    table = (data.groupby("month")["is_hit"].mean() * 100).round(0).reset_index()
    table["mois"] = table["month"].map(saisons)
    figure = px.bar(table, x="mois", y="is_hit", text="is_hit", color="is_hit",
                    color_continuous_scale="RdYlGn",
                    labels={"is_hit": "% de films rentables", "mois": ""},
                    title="Quand sortir un film")
    figure.update_traces(texttemplate="%{text} %", textposition="outside")
    figure.update_layout(height=420, coloraxis_showscale=False)
    return figure


def effet_saga(data: pd.DataFrame):
    table = (data.groupby("is_franchise")["is_hit"].mean() * 100).round(0).reset_index()
    table["type"] = np.where(table["is_franchise"] == 1, "suite ou univers existant", "film isolé")
    figure = px.bar(table, x="type", y="is_hit", text="is_hit", color="type",
                    color_discrete_map={"film isolé": "#C44E52", "suite ou univers existant": "#55A868"},
                    labels={"is_hit": "% de films rentables", "type": ""},
                    title="L'effet le plus fort du catalogue : appartenir à une saga")
    figure.update_traces(texttemplate="%{text} %", textposition="outside")
    figure.update_layout(height=420, showlegend=False)
    return figure


def simulateur_budget(grille: pd.DataFrame, seuil_pourcent: float = 70):
    """Probabilité selon le budget, avec un curseur sur le genre."""
    figure = px.line(grille.sort_values(["genre", "type de film", "budget"]),
                     x="budget", y="probabilite", color="type de film", markers=True,
                     animation_frame="genre", range_y=[0, 100],
                     color_discrete_map={"film isolé": "#C44E52", "suite ou univers existant": "#55A868"},
                     labels={"budget": "budget (millions de dollars 2023)",
                             "probabilite": "% de chances d'être rentable"},
                     title="Simulateur : faites varier le genre et suivez l'effet du budget")
    figure.add_hline(y=seuil_pourcent, line_dash="dash", annotation_text="seuil de recommandation")
    figure.update_layout(height=520)
    return figure


def simulateur_genre(grille: pd.DataFrame, seuil_pourcent: float = 70):
    """Classement des genres, avec un curseur sur le budget."""
    table = grille.sort_values(["budget", "probabilite"], ascending=[True, False])
    figure = px.bar(table, x="probabilite", y="genre", color="type de film", barmode="group",
                    animation_frame="budget", range_x=[0, 100], orientation="h",
                    color_discrete_map={"film isolé": "#C44E52", "suite ou univers existant": "#55A868"},
                    labels={"probabilite": "% de chances d'être rentable", "genre": ""},
                    title="Simulateur : faites varier le budget et comparez les genres")
    figure.add_vline(x=seuil_pourcent, line_dash="dash", annotation_text="seuil")
    figure.update_layout(height=560)
    return figure
