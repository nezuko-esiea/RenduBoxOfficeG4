"""Figures techniques du modèle de succès (Matplotlib et Seaborn)."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

VERT, ROUGE, BLEU, ORANGE, GRIS = "#55A868", "#C44E52", "#4C72B0", "#DD8452", "#8C8C8C"


def appliquer_style() -> None:
    sns.set_theme(style="whitegrid", palette="deep")
    plt.rcParams["figure.dpi"] = 110


def distribution_cible(data: pd.DataFrame, seuil_roi: float = 2.5):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(data["log_roi"], bins=60, ax=axes[0], color=BLEU)
    axes[0].axvline(np.log(seuil_roi), color=ROUGE, linestyle="--", label=f"seuil ROI = {seuil_roi}")
    axes[0].set(xlabel="log(ROI)", ylabel="films", title="Distribution du ROI (échelle logarithmique)")
    axes[0].legend()

    part = data["is_hit"].value_counts(normalize=True).sort_index() * 100
    axes[1].bar(["non rentable", "rentable"], part.values, color=[ROUGE, VERT])
    axes[1].set(ylabel="% des films", title=f"Équilibre des classes ({len(data):,} films)")
    for position, valeur in enumerate(part.values):
        axes[1].text(position, valeur + 1, f"{valeur:.1f} %", ha="center")
    plt.tight_layout()
    return fig


def seuil_rentabilite(resultat: pd.DataFrame, grille: pd.DataFrame, seuil_roi: float = 2.5):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), gridspec_kw={"width_ratios": [1.1, 1]})
    valeurs = resultat["résultat du studio"]
    etiquettes = [f"ROI {multiple:g}" for multiple in resultat.index]
    axes[0].bar(etiquettes, valeurs, color=[VERT if v > 0 else GRIS if v == 0 else ROUGE for v in valeurs])
    axes[0].axhline(0, color="black", linewidth=1)
    budget = resultat["production"].iloc[0]
    axes[0].set(ylabel="résultat du studio (M$)", ylim=(valeurs.min() - 25, valeurs.max() + 25),
                title=f"Film à {budget:.0f} M$ : ce qu'il reste au studio après les salles")
    for position, valeur in enumerate(valeurs):
        axes[0].text(position, valeur + (4 if valeur >= 0 else -4),
                     f"{valeur:+.0f} M$" if valeur else "0 : point mort",
                     ha="center", va="bottom" if valeur >= 0 else "top", fontweight="bold")

    sns.heatmap(grille, annot=True, fmt=".2f", cmap="YlOrRd", cbar=False, linewidths=1, ax=axes[1],
                vmin=1, vmax=4, annot_kws={"fontsize": 13, "fontweight": "bold"})
    axes[1].set(title="ROI à atteindre pour couvrir les coûts, selon les hypothèses")
    axes[1].set_xticklabels([nom.replace(" = ", "\n").replace(" du budget", "\ndu budget") for nom in grille.columns],
                            rotation=0, fontsize=9)
    axes[1].set_yticklabels([nom.replace("le studio touche ", "studio : ").replace(" des recettes", "\ndes recettes")
                             for nom in grille.index], rotation=0, fontsize=9)
    plt.tight_layout()
    return fig


def exploration(data: pd.DataFrame):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    quintiles = pd.qcut(data["log_budget"], 5)
    budget = data.groupby(quintiles, observed=True)["is_hit"].mean() * 100
    etiquettes = [f"{np.exp(i.left)/1e6:.0f}-{np.exp(i.right)/1e6:.0f} M$" for i in budget.index]
    sns.barplot(x=etiquettes, y=budget.values, ax=axes[0, 0], color=BLEU)
    axes[0, 0].set(title="Budget : une relation en U", ylabel="% de films rentables", xlabel="budget (dollars 2023)")
    axes[0, 0].tick_params(axis="x", rotation=20)

    franchise = data.groupby("is_franchise")["is_hit"].mean() * 100
    sns.barplot(x=["film isolé", "franchise"], y=franchise.values, ax=axes[0, 1], color=VERT)
    axes[0, 1].set(title="Appartenance à une saga", ylabel="% de films rentables")

    genres = [c for c in data.columns if c.startswith("genre_")]
    taux = {g[6:]: data.loc[data[g] == 1, "is_hit"].mean() * 100 for g in genres if data[g].sum() >= 150}
    taux = pd.Series(taux).sort_values(ascending=False)
    sns.barplot(x=taux.values, y=taux.index, ax=axes[1, 0], color=BLEU)
    axes[1, 0].set(title="Genre", xlabel="% de films rentables")

    saison = data.groupby("month")["is_hit"].mean() * 100
    sns.lineplot(x=saison.index, y=saison.values, marker="o", ax=axes[1, 1], color=ROUGE)
    axes[1, 1].set(title="Mois de sortie", xlabel="mois", ylabel="% de films rentables", xticks=range(1, 13))
    plt.tight_layout()
    return fig


def banc_essai(banc: pd.DataFrame, scores_par_bloc: dict):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    classement = banc["AUC"].sort_values()
    axes[0].barh(classement.index, classement.values, xerr=banc.loc[classement.index, "AUC std"],
                 color=[VERT if nom == banc.index[0] else GRIS for nom in classement.index], capsize=4)
    axes[0].axvline(0.5, color=ROUGE, linestyle="--", label="hasard")
    axes[0].set(xlim=(0.45, 0.80), xlabel="AUC en validation croisée",
                title="Familles comparées avant tout réglage")
    axes[0].legend(loc="lower right")
    for nom, valeur in classement.items():
        axes[0].text(valeur + 0.008, nom, f"{valeur:.3f}", va="center")

    ordre = banc.index.tolist()
    sns.boxplot(data=[scores_par_bloc[nom] for nom in ordre], orient="h", color=BLEU, ax=axes[1])
    axes[1].set_yticks(range(len(ordre)), ordre)
    axes[1].set(xlabel="AUC par bloc de validation croisée", title="Dispersion entre blocs")
    plt.tight_layout()
    return fig


def particularites(resultat: dict):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    budget = resultat["budget"] * 100
    positions = range(len(budget))
    axes[0].plot(positions, budget["observé"], marker="o", color="black", linewidth=2.5, label="observé")
    for nom, couleur in [("régression logistique", ORANGE), ("forêt aléatoire", GRIS), ("gradient boosting", VERT)]:
        axes[0].plot(positions, budget[nom], marker="o", color=couleur, label=nom)
    axes[0].set_xticks(positions, [f"{valeur:.0f}" for valeur in budget.index])
    axes[0].set(xlabel="budget médian de la tranche (M$, dollars 2023)", ylabel="% de films rentables",
                title="Qui reproduit la forme en U du budget ?")
    axes[0].legend()

    manquants = resultat["manquants"]
    hauteur = 0.38
    rangs = np.arange(len(manquants))
    axes[1].barh(rangs + hauteur / 2, manquants["films complets"], hauteur, color=BLEU, label="films complets")
    axes[1].barh(rangs - hauteur / 2, manquants["films avec valeur manquante"], hauteur, color=ORANGE,
                 label="films avec valeur manquante")
    for rang, (complet, incomplet) in enumerate(zip(manquants["films complets"],
                                                   manquants["films avec valeur manquante"])):
        axes[1].text(complet + 0.003, rang + hauteur / 2, f"{complet:.3f}", va="center")
        axes[1].text(incomplet + 0.003, rang - hauteur / 2, f"{incomplet:.3f}", va="center")
    axes[1].set_yticks(rangs, manquants.index)
    axes[1].set(xlim=(0.6, 0.8), xlabel="AUC en validation croisée",
                title="Qui résiste aux valeurs manquantes ?")
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
    plt.tight_layout()
    return fig


def apport(resultat: dict):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), gridspec_kw={"width_ratios": [1, 1.15, 1]})
    information = resultat["information"]
    axes[0].bar(range(len(information)), information.values,
                color=[GRIS] * (len(information) - 1) + [VERT])
    axes[0].axhline(0.5, color=ROUGE, linestyle="--", label="hasard")
    axes[0].set_xticks(range(len(information)), [nom.replace(" (", "\n(") for nom in information.index], fontsize=9)
    axes[0].set(ylim=(0.45, 0.8), ylabel="AUC sur le test", title="Aucune variable seule ne suffit")
    axes[0].legend(loc="upper left")
    for position, valeur in enumerate(information.values):
        axes[0].text(position, valeur + 0.006, f"{valeur:.3f}", ha="center")

    regles = resultat["regles"]
    base = regles["precision"].iloc[0] * 100
    choix = regles.iloc[1:]
    axes[1].bar(range(len(choix)), choix["precision"] * 100, color=[GRIS] * (len(choix) - 1) + [VERT])
    axes[1].axhline(base, color=ROUGE, linestyle="--", label=f"tout le catalogue : {base:.0f} %")
    axes[1].set_xticks(range(len(choix)), [nom.replace(" : ", "\n") for nom in choix.index], fontsize=9)
    axes[1].set(ylim=(0, 90), ylabel="% de films rentables parmi les retenus", title="Le modèle face aux règles simples")
    axes[1].legend(loc="upper left")
    for position, (films, valeur) in enumerate(zip(choix["films retenus"], choix["precision"] * 100)):
        axes[1].text(position, valeur + 1.5, f"{valeur:.0f} %\n{films} films", ha="center")

    sagas = resultat["sagas"]
    axes[2].bar(range(len(sagas)), sagas["taux_reel"] * 100, color=BLEU)
    axes[2].set_xticks(range(len(sagas)), sagas.index, fontsize=9)
    axes[2].set(ylim=(0, 90), xlabel="probabilité annoncée par le modèle", ylabel="% de sagas rentables",
                title="Toutes les sagas ne se valent pas")
    for position, (films, valeur) in enumerate(zip(sagas["films"], sagas["taux_reel"] * 100)):
        axes[2].text(position, valeur + 1.5, f"{valeur:.0f} %\n{films} films", ha="center")
    plt.tight_layout()
    return fig


def acp(resultat: dict):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    variance = resultat["variance_cumulee"]
    axes[0].plot(range(1, len(variance) + 1), variance * 100, marker=".", color=BLEU)
    axes[0].axhline(95, color=ROUGE, linestyle="--", label="95 % de variance")
    axes[0].set(xlabel="nombre de composantes", ylabel="variance cumulée (%)", title="Variance expliquée par l'ACP")
    axes[0].legend()

    valeurs = [resultat["auc_sans_acp"], resultat["auc_avec_acp"]]
    axes[1].bar(["sans ACP", "avec ACP (95 %)"], valeurs, color=[VERT, ROUGE])
    axes[1].set(ylim=(0.6, 0.8), ylabel="AUC en validation croisée", title="Effet de l'ACP")
    for position, valeur in enumerate(valeurs):
        axes[1].text(position, valeur + 0.005, f"{valeur:.3f}", ha="center")
    plt.tight_layout()
    return fig


def apprentissage(courbe: dict):
    fig = plt.figure(figsize=(7, 4))
    validation = courbe["validation"].mean(axis=1)
    plt.plot(courbe["tailles"], courbe["entrainement"].mean(axis=1), marker="o", label="entraînement")
    plt.plot(courbe["tailles"], validation, marker="o", label="validation")
    plt.fill_between(courbe["tailles"], validation - courbe["validation"].std(axis=1),
                     validation + courbe["validation"].std(axis=1), alpha=0.15)
    plt.xlabel("nombre de films d'entraînement")
    plt.ylabel("AUC")
    plt.title("Courbe d'apprentissage")
    plt.legend()
    plt.tight_layout()
    return fig


def perte(modele):
    fig = plt.figure(figsize=(7, 4))
    plt.plot(-modele.train_score_, label="entraînement")
    plt.plot(-modele.validation_score_, label="validation")
    plt.axvline(modele.n_iter_ - 1, color=GRIS, linestyle="--", label=f"arrêt anticipé : {modele.n_iter_} arbres")
    plt.xlabel("nombre d'arbres")
    plt.ylabel("log loss")
    plt.title("Courbe de perte")
    plt.legend()
    plt.tight_layout()
    return fig


def roc(reel, probabilites):
    fig = plt.figure(figsize=(6, 4.5))
    RocCurveDisplay.from_predictions(reel, probabilites, name="gradient boosting réglé", ax=plt.gca())
    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.title("Courbe ROC sur le test temporel")
    plt.tight_layout()
    return fig


def seuils_et_calibration(arbitrage: pd.DataFrame, calibration: pd.DataFrame, seuil_retenu: float):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(arbitrage.index, arbitrage["precision"], marker="o", label="precision")
    axes[0].plot(arbitrage.index, arbitrage["recall"], marker="o", label="recall")
    axes[0].axvline(seuil_retenu, color=GRIS, linestyle="--", label="seuil retenu")
    axes[0].set(xlabel="seuil de décision", title="Arbitrage precision / recall")
    axes[0].legend()

    axes[1].plot([0, 1], [0, 1], "k--", linewidth=1, label="calibration parfaite")
    axes[1].plot(calibration["annonce"], calibration["observe"], marker="o", color=BLEU, label="modèle")
    axes[1].set(xlabel="probabilité annoncée", ylabel="taux observé", title="Courbe de calibration")
    axes[1].legend()
    plt.tight_layout()
    return fig


def confusion(reel, decision, seuil: float):
    fig = plt.figure(figsize=(5.5, 4.5))
    ConfusionMatrixDisplay.from_predictions(reel, decision, ax=plt.gca(), colorbar=False,
                                            display_labels=["non recommandé", "recommandé"], cmap="Blues")
    plt.title(f"Matrice de confusion (seuil {seuil})")
    plt.tight_layout()
    return fig


def precision_par_periode(table: pd.DataFrame, taux_de_base: float):
    fig = plt.figure(figsize=(8, 4.5))
    graphique = sns.barplot(x=table.index, y=table["precision"], hue=table.index,
                            palette=[VERT, ROUGE, BLEU], legend=False)
    plt.axhline(taux_de_base, color="black", linestyle="--",
                label=f"taux de base du catalogue ({100 * taux_de_base:.0f} %)")
    plt.ylabel("precision des recommandations")
    plt.xlabel("")
    plt.title("La seule période de décrochage est celle de la fermeture des salles")
    plt.legend()
    for position, valeur in enumerate(table["precision"]):
        graphique.text(position, valeur + 0.02, f"{100 * valeur:.0f} %", ha="center", fontweight="bold")
    plt.tight_layout()
    return fig


def importance(classement: pd.Series, premieres: int = 15):
    fig = plt.figure(figsize=(8, 5))
    extrait = classement.head(premieres)
    couleurs = [ROUGE if rang < 5 else BLEU for rang in range(len(extrait))]
    sns.barplot(x=extrait.values, y=extrait.index, hue=extrait.index, palette=couleurs, legend=False)
    plt.xlabel("perte d'AUC quand la variable est mélangée")
    plt.title("Importance par permutation (les 5 premières en rouge)")
    plt.tight_layout()
    return fig


def coefficients(serie: pd.Series, extremes: int = 8):
    fig = plt.figure(figsize=(8, 5))
    selection = pd.concat([serie.nlargest(extremes), serie.nsmallest(extremes)]).sort_values()
    sns.barplot(x=selection.values, y=selection.index, hue=selection.index,
                palette=[ROUGE if v < 0 else VERT for v in selection.values], legend=False)
    plt.axvline(0, color="black", linewidth=1)
    plt.xlabel("coefficient : effet sur la probabilité de rentabilité")
    plt.title("Ce qui augmente et ce qui diminue les chances de succès")
    plt.tight_layout()
    return fig


def controle_fuite(table: pd.DataFrame):
    fig = plt.figure(figsize=(7, 4))
    etiquettes = ["34 variables\npré-production", "+ votes, popularité", "+ recettes"]
    plt.bar(etiquettes, table["AUC"].values, color=[VERT, ORANGE, ROUGE])
    plt.ylim(0.5, 1.05)
    plt.ylabel("AUC")
    plt.title("Ce que coûterait une fuite de données")
    for position, valeur in enumerate(table["AUC"].values):
        plt.text(position, valeur + 0.01, f"{valeur:.3f}", ha="center", fontweight="bold")
    plt.tight_layout()
    return fig


def gain_seuil(arbitrage: pd.DataFrame, seuil_initial: float = 0.5, seuil_retenu: float = 0.7):
    lignes = arbitrage.loc[[seuil_initial, seuil_retenu]]
    etiquettes = [f"seuil {seuil_initial}", f"seuil {seuil_retenu}"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].bar(etiquettes, lignes["precision"] * 100, color=[GRIS, VERT])
    axes[0].set(ylabel="precision (%)", title="Fiabilité des films recommandés")
    for position, valeur in enumerate(lignes["precision"] * 100):
        axes[0].text(position, valeur + 1, f"{valeur:.0f} %", ha="center", fontweight="bold")

    axes[1].bar(etiquettes, lignes["faux positifs"], color=[ROUGE, VERT])
    axes[1].set(ylabel="films", title="Faux positifs : flops annoncés rentables")
    for position, valeur in enumerate(lignes["faux positifs"]):
        axes[1].text(position, valeur + 3, f"{int(valeur)}", ha="center", fontweight="bold")
    plt.tight_layout()
    return fig


def zones_decision(probabilites, reel, seuil_rejet: float = 0.3, seuil_recommandation: float = 0.7):
    """Distribution des probabilités prédites, colorée par la réalité, avec les trois zones."""
    reel = np.asarray(reel)
    fig, axe = plt.subplots(figsize=(11, 5))
    intervalles = np.linspace(0, 1, 26)
    axe.hist([probabilites[reel == 0], probabilites[reel == 1]], bins=intervalles, stacked=True,
             color=[ROUGE, VERT], label=["non rentable", "rentable"], edgecolor="white", linewidth=0.4)

    bornes = [(0, seuil_rejet, "écarter", ROUGE), (seuil_rejet, seuil_recommandation, "examiner", ORANGE),
              (seuil_recommandation, 1.0, "recommander", VERT)]
    hauteur = axe.get_ylim()[1]
    for debut, fin, nom, couleur in bornes:
        dans_zone = (probabilites >= debut) & (probabilites < fin if fin < 1 else probabilites <= fin)
        films, taux = int(dans_zone.sum()), reel[dans_zone].mean() * 100
        axe.axvspan(debut, fin, color=couleur, alpha=0.08)
        axe.text((debut + fin) / 2, hauteur * 0.97, nom.upper(), ha="center", va="top",
                 fontweight="bold", color=couleur)
        axe.text((debut + fin) / 2, hauteur * 0.88, f"{films} films\n{taux:.0f} % rentables",
                 ha="center", va="top", color=couleur)

    for seuil in (seuil_rejet, seuil_recommandation):
        axe.axvline(seuil, color="black", linestyle="--", linewidth=1)
    axe.set(xlabel="probabilité de rentabilité estimée par le modèle", ylabel="films",
            title="Pourquoi une zone d'abstention : au milieu, le modèle ne tranche pas")
    axe.legend(loc="upper right", bbox_to_anchor=(1, 0.75))
    plt.tight_layout()
    return fig


def probabilite_roi(diagnostic: pd.DataFrame, seuil_roi: float = 2.5, seuil_rejet: float = 0.3,
                    seuil_recommandation: float = 0.7, tranches: int = 10):
    """ROI réel de chaque film du test en fonction de la probabilité annoncée par le modèle."""
    fig, axe = plt.subplots(figsize=(11, 5.5))
    for rentable, couleur, nom in [(0, ROUGE, "non rentable"), (1, VERT, "rentable")]:
        groupe = diagnostic[diagnostic["rentable"] == rentable]
        axe.scatter(groupe["probabilite"], groupe["roi"], s=14, alpha=0.45, color=couleur, label=nom)

    tranche = pd.qcut(diagnostic["probabilite"], tranches, duplicates="drop")
    mediane = diagnostic.groupby(tranche, observed=True).agg(probabilite=("probabilite", "median"), roi=("roi", "median"))
    axe.plot(mediane["probabilite"], mediane["roi"], color="black", marker="o", linewidth=2.5,
             label="ROI médian par tranche de probabilité")

    axe.axhline(seuil_roi, color=GRIS, linestyle="--", label=f"seuil de rentabilité : ROI = {seuil_roi}")
    for seuil in (seuil_rejet, seuil_recommandation):
        axe.axvline(seuil, color="black", linestyle=":", linewidth=1)
    axe.set_yscale("log")
    axe.set(xlim=(0, 1), xlabel="probabilité de rentabilité annoncée par le modèle",
            ylabel="ROI réel (échelle logarithmique)",
            title="Plus la probabilité annoncée est élevée, plus le ROI réel l'est aussi")
    axe.legend(loc="upper left")
    plt.tight_layout()
    return fig


def sensibilite_seuil(table: pd.DataFrame, seuil_retenu: float = 2.5):
    """AUC obtenue selon le seuil de rentabilité choisi pour définir la cible."""
    fig = plt.figure(figsize=(8, 4))
    plt.plot(table.index, table["AUC test"], marker="o", color=BLEU)
    plt.axvline(seuil_retenu, color=VERT, linestyle="--", label=f"seuil retenu : {seuil_retenu}")
    plt.ylim(0.60, 0.85)
    plt.xlabel("seuil de rentabilité retenu pour définir la cible (ROI)")
    plt.ylabel("AUC en test")
    plt.title("Le choix du seuil ne change pas ce que le modèle apprend")
    for abscisse, valeur in zip(table.index, table["AUC test"]):
        plt.text(abscisse, valeur + 0.012, f"{valeur:.3f}", ha="center")
    plt.legend()
    plt.tight_layout()
    return fig
