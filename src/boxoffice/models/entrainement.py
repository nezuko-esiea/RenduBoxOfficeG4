"""Expériences du modèle de succès : préparation, banc d'essai, réglage, évaluation, seuils."""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, brier_score_loss, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold, cross_val_predict,
                                     cross_val_score, cross_validate, learning_curve)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from boxoffice import config

CATEGORIELLES = ["saison", "langue", "pays", "studio"]
SEUIL_RECOMMANDATION = 0.7
SEUIL_REJET = 0.3
ANNEE_TEST = 2018
GRILLE = {
    "learning_rate": [0.03, 0.05, 0.1, 0.2],
    "max_leaf_nodes": [15, 31, 63],
    "min_samples_leaf": [10, 20, 40],
    "l2_regularization": [0.0, 0.1, 1.0, 10.0],
    "max_iter": [200, 400],
}
MOIS_SAISON = {"hiver": 1, "printemps": 4, "ete": 7, "rentree": 9, "automne": 10, "fin_annee": 12}


def preparer(data: pd.DataFrame, variables: list[str]) -> dict:
    """Matrice d'entrée, cible et découpage temporel entraînement / test."""
    numeriques = [c for c in variables if c not in CATEGORIELLES]
    X = data[variables].copy()
    for colonne in variables:
        X[colonne] = (X[colonne].astype("category").cat.codes.astype("float64")
                      if colonne in CATEGORIELLES else X[colonne].astype("float64"))
    y = data["is_hit"].astype(int)
    i_train = data.index[data["year"] < ANNEE_TEST]
    i_test = data.index[data["year"] >= ANNEE_TEST]
    return {"X": X, "y": y, "i_train": i_train, "i_test": i_test, "numeriques": numeriques,
            "indices_categoriels": [variables.index(c) for c in CATEGORIELLES],
            "variables": list(variables)}


def preparation_colonnes(numeriques: list[str]) -> ColumnTransformer:
    return ColumnTransformer([
        ("num", Pipeline([("imputation", SimpleImputer(strategy="median")), ("echelle", StandardScaler())]), numeriques),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORIELLES),
    ])


def candidats(jeu: dict) -> dict:
    preparation = preparation_colonnes(jeu["numeriques"])
    return {
        "régression logistique": Pipeline([("prep", preparation), ("modele", LogisticRegression(max_iter=2000))]),
        "arbre de décision": Pipeline([("imputation", SimpleImputer(strategy="median")),
                                       ("modele", DecisionTreeClassifier(max_depth=6, random_state=42))]),
        "forêt aléatoire": Pipeline([("imputation", SimpleImputer(strategy="median")),
                                     ("modele", RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1))]),
        "gradient boosting": HistGradientBoostingClassifier(
            categorical_features=jeu["indices_categoriels"], random_state=42),
    }


def validation_croisee(blocs: int = 5) -> StratifiedKFold:
    return StratifiedKFold(blocs, shuffle=True, random_state=42)


def banc_essai(jeu: dict, modeles: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Compare les familles de modèles en validation croisée sur le jeu d'entraînement."""
    modeles = modeles or candidats(jeu)
    cv = validation_croisee()
    resultats, scores_par_bloc = [], {}
    for nom, modele in modeles.items():
        mesures = cross_validate(modele, jeu["X"].loc[jeu["i_train"]], jeu["y"][jeu["i_train"]],
                                 cv=cv, scoring=["roc_auc", "f1", "accuracy"], n_jobs=1)
        scores_par_bloc[nom] = mesures["test_roc_auc"]
        resultats.append({"modèle": nom,
                          "AUC": mesures["test_roc_auc"].mean(),
                          "AUC std": mesures["test_roc_auc"].std(),
                          "F1": mesures["test_f1"].mean(),
                          "accuracy": mesures["test_accuracy"].mean(),
                          "fit time (s)": mesures["fit_time"].mean()})
    banc = pd.DataFrame(resultats).set_index("modèle").sort_values("AUC", ascending=False)
    return banc, scores_par_bloc


def sensibilite_seuil(data: pd.DataFrame, jeu: dict, seuils=(1.0, 1.5, 2.0, 2.5, 3.0), retenus: int = 120) -> pd.DataFrame:
    """Réentraîne le modèle pour plusieurs définitions du succès et compare ce qu'il apprend et ce qu'il recommande."""
    roi = np.exp(data["log_roi"])
    train, test = jeu["X"].loc[jeu["i_train"]], jeu["X"].loc[jeu["i_test"]]
    roi_test = roi[jeu["i_test"]].values
    lignes = []
    for seuil in seuils:
        cible = (roi >= seuil).astype(int)
        modele = HistGradientBoostingClassifier(categorical_features=jeu["indices_categoriels"], random_state=42)
        modele.fit(train, cible[jeu["i_train"]])
        probabilites = modele.predict_proba(test)[:, 1]
        importance = permutation_importance(modele, test, cible[jeu["i_test"]], scoring="roc_auc",
                                            n_repeats=5, random_state=42)
        premieres = pd.Series(importance.importances_mean, index=jeu["variables"]).nlargest(3).index
        meilleurs = np.argsort(-probabilites)[:retenus]
        lignes.append({"seuil ROI": seuil,
                       "part de succès %": 100 * cible.mean(),
                       "AUC test": roc_auc_score(cible[jeu["i_test"]], probabilites),
                       "3 premières variables": ", ".join(premieres),
                       f"ROI médian des {retenus} films les mieux classés": np.median(roi_test[meilleurs]),
                       "dont films à perte %": 100 * (roi_test[meilleurs] < 1).mean()})
    return pd.DataFrame(lignes).set_index("seuil ROI")


def particularites(jeu: dict, tranches: int = 10) -> dict:
    """Ce qui distingue les familles : forme de l'effet budget et tenue face aux valeurs manquantes."""
    X, y = jeu["X"].loc[jeu["i_train"]], jeu["y"][jeu["i_train"]]
    cv = validation_croisee()
    probabilites = {nom: cross_val_predict(modele, X, y, cv=cv, method="predict_proba")[:, 1]
                    for nom, modele in candidats(jeu).items()}

    tranche = pd.qcut(X["log_budget"], tranches, labels=False)
    budget = pd.DataFrame({"budget médian (M$)": np.exp(X["log_budget"]).groupby(tranche).median() / 1e6,
                           "observé": y.groupby(tranche).mean()})
    for nom, valeurs in probabilites.items():
        budget[nom] = pd.Series(valeurs, index=X.index).groupby(tranche).mean()

    incomplet = X.isna().any(axis=1).values
    manquants = pd.DataFrame([
        {"modèle": nom,
         "films complets": roc_auc_score(y[~incomplet], valeurs[~incomplet]),
         "films avec valeur manquante": roc_auc_score(y[incomplet], valeurs[incomplet])}
        for nom, valeurs in probabilites.items()
    ]).set_index("modèle")
    manquants["écart"] = manquants["films avec valeur manquante"] - manquants["films complets"]

    seuls = {"régression logistique": Pipeline([("echelle", StandardScaler()), ("modele", LogisticRegression())]),
             "gradient boosting": HistGradientBoostingClassifier(random_state=42)}
    budget_seul = {nom: cross_val_score(modele, X[["log_budget"]], y, cv=cv, scoring="roc_auc").mean()
                   for nom, modele in seuls.items()}
    return {"budget": budget.set_index("budget médian (M$)"), "manquants": manquants, "budget_seul": budget_seul,
            "part_incomplets": incomplet.mean(),
            "taux_manquants": X.isna().mean().loc[lambda s: s > 0.001].sort_values(ascending=False)}


def comparer_acp(jeu: dict, variance_cible: float = 0.95) -> dict:
    """Mesure l'effet d'une réduction de dimension sur la régression logistique."""
    preparation = preparation_colonnes(jeu["numeriques"])
    cv = validation_croisee()
    sans = cross_val_score(Pipeline([("prep", preparation), ("modele", LogisticRegression(max_iter=2000))]),
                           jeu["X"].loc[jeu["i_train"]], jeu["y"][jeu["i_train"]], cv=cv, scoring="roc_auc")
    avec = cross_val_score(Pipeline([("prep", preparation), ("acp", PCA(n_components=variance_cible)),
                                     ("modele", LogisticRegression(max_iter=2000))]),
                           jeu["X"].loc[jeu["i_train"]], jeu["y"][jeu["i_train"]], cv=cv, scoring="roc_auc")
    analyse = Pipeline([("prep", preparation), ("acp", PCA())]).fit(jeu["X"].loc[jeu["i_train"]])
    variance = np.cumsum(analyse.named_steps["acp"].explained_variance_ratio_)
    return {"auc_sans_acp": sans.mean(), "auc_avec_acp": avec.mean(), "variance_cumulee": variance,
            "composantes_95": int((variance < variance_cible).sum() + 1), "composantes_totales": len(variance)}


def regler(jeu: dict, essais: int = 25) -> RandomizedSearchCV:
    """Recherche aléatoire d'hyperparamètres du gradient boosting."""
    recherche = RandomizedSearchCV(
        HistGradientBoostingClassifier(categorical_features=jeu["indices_categoriels"],
                                       early_stopping=True, random_state=42),
        GRILLE, n_iter=essais, scoring="roc_auc", cv=validation_croisee(), random_state=42, n_jobs=-1)
    recherche.fit(jeu["X"].loc[jeu["i_train"]], jeu["y"][jeu["i_train"]])
    return recherche


def courbe_apprentissage(modele, jeu: dict, blocs: int = 3) -> dict:
    tailles, score_train, score_val = learning_curve(
        modele, jeu["X"].loc[jeu["i_train"]], jeu["y"][jeu["i_train"]], cv=blocs, scoring="roc_auc",
        train_sizes=np.linspace(0.1, 1.0, 6), random_state=42)
    return {"tailles": tailles, "entrainement": score_train, "validation": score_val}


def suivi_perte(jeu: dict, arbres_max: int = 300) -> HistGradientBoostingClassifier:
    modele = HistGradientBoostingClassifier(
        categorical_features=jeu["indices_categoriels"], early_stopping=True, scoring="loss",
        validation_fraction=0.15, max_iter=arbres_max, random_state=42)
    return modele.fit(jeu["X"].loc[jeu["i_train"]], jeu["y"][jeu["i_train"]])


def predire(modele, jeu: dict) -> np.ndarray:
    modele.fit(jeu["X"].loc[jeu["i_train"]], jeu["y"][jeu["i_train"]])
    return modele.predict_proba(jeu["X"].loc[jeu["i_test"]])[:, 1]


def evaluer(jeu: dict, probabilites: np.ndarray, seuil: float = SEUIL_RECOMMANDATION) -> dict:
    reel = jeu["y"][jeu["i_test"]]
    decision = (probabilites >= seuil).astype(int)
    return {"AUC": roc_auc_score(reel, probabilites),
            "accuracy": accuracy_score(reel, decision),
            "precision": precision_score(reel, decision, zero_division=0),
            "recall": recall_score(reel, decision, zero_division=0),
            "F1": f1_score(reel, decision, zero_division=0),
            "Brier": brier_score_loss(reel, probabilites)}


def arbitrage_seuils(jeu: dict, probabilites: np.ndarray,
                     seuils=(0.3, 0.4, 0.5, 0.6, 0.7, 0.8)) -> pd.DataFrame:
    reel = jeu["y"][jeu["i_test"]]
    lignes = []
    for seuil in seuils:
        _, faux_positifs, faux_negatifs, vrais_positifs = confusion_matrix(reel, (probabilites >= seuil).astype(int)).ravel()
        lignes.append({"seuil": seuil,
                       "films recommandés": vrais_positifs + faux_positifs,
                       "vrais positifs": vrais_positifs,
                       "faux positifs": faux_positifs,
                       "succès manqués": faux_negatifs,
                       "precision": vrais_positifs / max(vrais_positifs + faux_positifs, 1),
                       "recall": vrais_positifs / (vrais_positifs + faux_negatifs)})
    return pd.DataFrame(lignes).set_index("seuil")


def calibration(jeu: dict, probabilites: np.ndarray) -> pd.DataFrame:
    tranches = pd.cut(probabilites, [0, 0.2, 0.4, 0.6, 0.8, 1.0])
    table = pd.DataFrame({"probabilite": probabilites, "reel": jeu["y"][jeu["i_test"]].values})
    return table.groupby(tranches, observed=True).agg(annonce=("probabilite", "mean"),
                                                      observe=("reel", "mean"),
                                                      films=("reel", "size"))


def zones(jeu: dict, probabilites: np.ndarray) -> pd.DataFrame:
    etiquettes = pd.cut(probabilites, [0, SEUIL_REJET, SEUIL_RECOMMANDATION, 1.0],
                        labels=["écarter", "examiner", "recommander"])
    table = pd.DataFrame({"zone": etiquettes, "rentable": jeu["y"][jeu["i_test"]].values})
    return table.groupby("zone", observed=True).agg(films=("rentable", "size"), taux_reel=("rentable", "mean"))


def diagnostic_erreurs(data: pd.DataFrame, jeu: dict, probabilites: np.ndarray) -> pd.DataFrame:
    test = data.loc[jeu["i_test"]]
    return pd.DataFrame({
        "film": test["title"].values,
        "annee": test["year"].values,
        "probabilite": probabilites,
        "rentable": jeu["y"][jeu["i_test"]].values,
        "roi": np.exp(test["log_roi"]).values,
        "budget_millions": np.exp(test["log_budget"]).values / 1e6,
    })


def causes_faux_positifs(diagnostic: pd.DataFrame, seuil: float = 0.5) -> pd.DataFrame:
    faux = diagnostic[(diagnostic["rentable"] == 0) & (diagnostic["probabilite"] >= seuil)]
    return pd.DataFrame([
        {"cause": "ROI entre 2 et 2,5 : seuil manqué de peu", "films": int(faux["roi"].between(2, 2.5).sum())},
        {"cause": "sorti en 2020 ou 2021 (fermeture des salles)", "films": int(faux["annee"].isin([2020, 2021]).sum())},
        {"cause": "budget TMDB supérieur à 400 M$ (valeur suspecte)", "films": int((faux["budget_millions"] > 400).sum())},
        {"cause": "ROI inférieur à 1 : véritable échec", "films": int((faux["roi"] < 1).sum())},
    ]).set_index("cause")


def precision_par_periode(diagnostic: pd.DataFrame, seuil: float = 0.5) -> pd.DataFrame:
    periodes = {"2018-2019": diagnostic["annee"].isin([2018, 2019]),
                "2020-2021 (COVID)": diagnostic["annee"].isin([2020, 2021]),
                "2022-2023": diagnostic["annee"] >= 2022}
    return pd.DataFrame([
        {"période": nom,
         "films recommandés": int((diagnostic.loc[masque, "probabilite"] >= seuil).sum()),
         "precision": diagnostic.loc[masque & (diagnostic["probabilite"] >= seuil), "rentable"].mean()}
        for nom, masque in periodes.items()
    ]).set_index("période")


def importance_variables(modele, jeu: dict, repetitions: int = 10) -> pd.Series:
    mesure = permutation_importance(modele, jeu["X"].loc[jeu["i_test"]], jeu["y"][jeu["i_test"]],
                                    scoring="roc_auc", n_repeats=repetitions, random_state=42)
    return pd.Series(mesure.importances_mean, index=jeu["variables"]).sort_values(ascending=False)


def explication_shap(modele, jeu: dict, reference: int = 100):
    """Valeurs SHAP par permutation, en probabilité : TreeExplainer ne lit pas les catégories natives du modèle."""
    import shap

    echantillon = shap.utils.sample(jeu["X"].loc[jeu["i_train"]], reference)
    fond = shap.maskers.Independent(echantillon, max_samples=reference)
    explicateur = shap.explainers.Permutation(lambda matrice: modele.predict_proba(matrice)[:, 1], fond,
                                              feature_names=jeu["variables"], seed=42)
    return explicateur(jeu["X"].loc[jeu["i_test"]], silent=True)


def sens_des_effets(explication, jeu: dict) -> pd.DataFrame:
    """Compare, variable par variable, le sens de l'effet selon SHAP et selon la régression logistique."""
    test = jeu["X"].loc[jeu["i_test"]]
    coefficients = coefficients_logistique(jeu)
    lignes = []
    for rang, nom in enumerate(jeu["variables"]):
        connu = test[nom].notna().values
        if nom in CATEGORIELLES or test.loc[connu, nom].nunique() < 2:
            continue
        with np.errstate(invalid="ignore", divide="ignore"):
            lien = np.corrcoef(test.loc[connu, nom], explication.values[connu, rang])[0, 1]
        lignes.append({"variable": nom,
                       "poids SHAP (points)": 100 * np.abs(explication.values[:, rang]).mean(),
                       "sens gradient boosting": "+" if lien > 0 else "-",
                       "sens régression logistique": "+" if coefficients[nom] > 0 else "-"})
    table = pd.DataFrame(lignes).set_index("variable").sort_values("poids SHAP (points)", ascending=False)
    table["accord"] = table["sens gradient boosting"] == table["sens régression logistique"]
    return table


def coefficients_logistique(jeu: dict) -> pd.Series:
    modele = Pipeline([("prep", preparation_colonnes(jeu["numeriques"])),
                       ("modele", LogisticRegression(max_iter=2000))])
    modele.fit(jeu["X"].loc[jeu["i_train"]], jeu["y"][jeu["i_train"]])
    noms = modele.named_steps["prep"].get_feature_names_out()
    return pd.Series(modele.named_steps["modele"].coef_[0], index=[n.split("__")[-1] for n in noms])


def controle_fuite(data: pd.DataFrame, jeu: dict) -> pd.DataFrame:
    """Compare l'AUC du modèle honnête à celle obtenue avec des variables postérieures à la sortie."""
    apres_sortie = pd.read_parquet(config.MOVIES_TABLE)[["movieId", "vote_count", "popularity", "revenue_2023"]]
    elargi = data.merge(apres_sortie, on="movieId")

    def auc_avec(colonnes):
        matrice = elargi[colonnes].copy()
        for colonne in colonnes:
            matrice[colonne] = (matrice[colonne].astype("category").cat.codes.astype("float64")
                                if colonne in CATEGORIELLES else matrice[colonne].astype("float64"))
        cible = elargi["is_hit"].astype(int)
        avant, apres = elargi["year"] < ANNEE_TEST, elargi["year"] >= ANNEE_TEST
        modele = HistGradientBoostingClassifier(
            categorical_features=[colonnes.index(c) for c in CATEGORIELLES], random_state=42)
        modele.fit(matrice[avant], cible[avant])
        return roc_auc_score(cible[apres], modele.predict_proba(matrice[apres])[:, 1])

    variables = jeu["variables"]
    return pd.DataFrame([
        {"variables utilisées": f"{len(variables)} variables pré-production (réglages par défaut)", "AUC": auc_avec(variables)},
        {"variables utilisées": "+ nombre de votes et popularité", "AUC": auc_avec(variables + ["vote_count", "popularity"])},
        {"variables utilisées": "+ recettes", "AUC": auc_avec(variables + ["revenue_2023"])},
    ]).set_index("variables utilisées")


def apport_modele(modele, data: pd.DataFrame, jeu: dict, probabilites: np.ndarray,
                  seuil: float = SEUIL_RECOMMANDATION, gros_budget: float = 100) -> dict:
    """Compare le modèle à des règles simples sur le test : information utilisée, précision, tri des sagas."""
    train, test = jeu["X"].loc[jeu["i_train"]], jeu["X"].loc[jeu["i_test"]]
    reel = jeu["y"][jeu["i_test"]].values
    roi = np.exp(data.loc[jeu["i_test"], "log_roi"].values)
    budget = np.exp(test["log_budget"].values) / 1e6
    saga = test["is_franchise"].values == 1

    information = {}
    for nom, colonnes in [("budget seul", ["log_budget"]), ("saga seule", ["is_franchise"]),
                          ("budget et saga", ["log_budget", "is_franchise"])]:
        reduit = clone(modele).set_params(categorical_features=None)
        reduit.fit(train[colonnes], jeu["y"][jeu["i_train"]])
        information[nom] = roc_auc_score(reel, reduit.predict_proba(test[colonnes])[:, 1])
    information[f"{len(jeu['variables'])} variables (modèle)"] = roc_auc_score(reel, probabilites)

    selections = {"tout le catalogue": np.ones(len(reel), dtype=bool),
                  f"règle : budget supérieur à {gros_budget:.0f} M$": budget > gros_budget,
                  "règle : toutes les sagas": saga,
                  f"modèle : probabilité au-dessus de {seuil}": probabilites >= seuil}
    regles = pd.DataFrame([
        {"sélection": nom, "films retenus": int(masque.sum()), "precision": reel[masque].mean(),
         "ROI médian": np.median(roi[masque]), "part de films à perte (ROI < 1)": (roi[masque] < 1).mean()}
        for nom, masque in selections.items()
    ]).set_index("sélection")

    zones_sagas = pd.cut(probabilites[saga], [0, SEUIL_REJET, 0.5, seuil, 1.0],
                         labels=[f"sous {SEUIL_REJET}", f"{SEUIL_REJET} à 0.5", f"0.5 à {seuil}", f"au-dessus de {seuil}"])
    sagas = pd.DataFrame({"zone": zones_sagas, "rentable": reel[saga]}).groupby("zone", observed=True).agg(
        films=("rentable", "size"), taux_reel=("rentable", "mean"))
    return {"information": pd.Series(information, name="AUC"), "regles": regles, "sagas": sagas,
            "sagas_recommandees": int((saga & (probabilites >= seuil)).sum())}


def _codes_categories(data: pd.DataFrame) -> dict:
    return {colonne: {valeur: rang for rang, valeur in enumerate(pd.Categorical(data[colonne]).categories)}
            for colonne in CATEGORIELLES}


def projet(data: pd.DataFrame, jeu: dict, nom: str, genre: str, budget_millions: float, franchise: int,
           saison: str, studio: str = "major", pays: str = "US", langue: str = "en",
           age: int = 13, duree: int = 110) -> pd.Series:
    """Construit un film fictif : variables médianes, sauf celles que l'on fait varier."""
    codes = _codes_categories(data)
    ligne = jeu["X"].loc[jeu["i_train"]].median()
    for colonne in [c for c in jeu["variables"] if c.startswith("genre_")]:
        ligne[colonne] = 0
    ligne[f"genre_{genre}"] = 1
    ligne["log_budget"] = np.log(budget_millions * 1e6)
    ligne["runtime"] = duree
    ligne["age_min"] = age
    ligne["is_franchise"] = franchise
    ligne["periode_covid"] = 0
    ligne["month"] = MOIS_SAISON[saison]
    for colonne, valeur in [("saison", saison), ("studio", studio), ("pays", pays), ("langue", langue)]:
        ligne[colonne] = codes[colonne][valeur]
    return ligne.rename(nom)


def simuler(modele, data: pd.DataFrame, jeu: dict, projets: list[dict]) -> pd.DataFrame:
    lignes = [projet(data, jeu, **specification) for specification in projets]
    matrice = pd.DataFrame(lignes).astype("float64")[jeu["variables"]]
    probabilites = modele.predict_proba(matrice)[:, 1]
    return pd.DataFrame({
        "probabilité de rentabilité": probabilites.round(3),
        "décision": np.where(probabilites >= SEUIL_RECOMMANDATION, "recommander",
                             np.where(probabilites < SEUIL_REJET, "écarter", "examiner")),
    }, index=[specification["nom"] for specification in projets]).sort_values(
        "probabilité de rentabilité", ascending=False)


def grille_genre_budget(modele, data: pd.DataFrame, jeu: dict, genres: list[str],
                        budgets=(10, 30, 80, 150)) -> pd.DataFrame:
    lignes, etiquettes = [], []
    for genre in genres:
        for budget in budgets:
            lignes.append(projet(data, jeu, f"{genre}|{budget}", genre, budget, 0, "ete"))
            etiquettes.append((genre, f"{budget} M$"))
    matrice = pd.DataFrame(lignes).astype("float64")[jeu["variables"]]
    probabilites = modele.predict_proba(matrice)[:, 1] * 100
    table = pd.DataFrame(etiquettes, columns=["genre", "budget"]).assign(probabilite=probabilites.round(0))
    return table.pivot(index="genre", columns="budget", values="probabilite")[[f"{b} M$" for b in budgets]]


def courbe_budget_saga(modele, data: pd.DataFrame, jeu: dict, genre: str = "Action",
                       budgets=(5, 10, 20, 40, 80, 120, 160, 200, 250)) -> pd.DataFrame:
    lignes = []
    for franchise, etiquette in [(0, "film isolé"), (1, "suite ou univers existant")]:
        for budget in budgets:
            ligne = projet(data, jeu, f"{etiquette}|{budget}", genre, budget, franchise, "ete")
            probabilite = modele.predict_proba(pd.DataFrame([ligne]).astype("float64")[jeu["variables"]])[0, 1]
            lignes.append({"type de film": etiquette, "budget": budget, "probabilité": round(100 * probabilite)})
    return pd.DataFrame(lignes)


def grille_simulation(modele, data: pd.DataFrame, jeu: dict, genres: list[str],
                      budgets=(5, 10, 20, 30, 50, 80, 120, 160, 200, 250), saison: str = "ete") -> pd.DataFrame:
    """Prédictions du modèle sur une grille de projets fictifs : genre x budget x appartenance à une saga."""
    lignes, etiquettes = [], []
    for genre in genres:
        for budget in budgets:
            for franchise, type_film in [(0, "film isolé"), (1, "suite ou univers existant")]:
                lignes.append(projet(data, jeu, f"{genre}|{budget}|{franchise}", genre, budget,
                                     franchise, saison))
                etiquettes.append({"genre": genre, "budget": budget, "type de film": type_film})
    matrice = pd.DataFrame(lignes).astype("float64")[jeu["variables"]]
    probabilites = modele.predict_proba(matrice)[:, 1] * 100
    return pd.DataFrame(etiquettes).assign(probabilite=probabilites.round(0))

def fiabilite_par_genre(data: pd.DataFrame, jeu: dict, probabilites: np.ndarray,
                        minimum: int = 30) -> pd.DataFrame:
    """Pour chaque genre : taux réel, probabilité moyenne annoncée, biais et Brier.

    Le biais vaut probabilité moyenne - taux réel. Positif, le modèle surestime ce genre ;
    négatif, il le sous-estime. Plus le Brier est faible, mieux le genre est prédit.
    """
    test = data.loc[jeu["i_test"]]
    reel = jeu["y"][jeu["i_test"]].to_numpy()
    lignes = []
    for colonne in [c for c in jeu["variables"] if c.startswith("genre_")]:
        masque = test[colonne].to_numpy() == 1
        if masque.sum() < minimum:
            continue
        annonce, observe = probabilites[masque], reel[masque]
        lignes.append({"type de film": colonne[6:], "films": int(masque.sum()),
                       "taux réel": observe.mean(), "probabilité moyenne": annonce.mean(),
                       "biais": annonce.mean() - observe.mean(), "Brier": brier_score_loss(observe, annonce)})
    return pd.DataFrame(lignes).set_index("type de film").sort_values("Brier")


def residus_par_genre(data: pd.DataFrame, jeu: dict, probabilites: np.ndarray,
                      minimum: int = 30) -> pd.DataFrame:
    """Un résidu par film et par genre : probabilité annoncée moins issue réelle (0 ou 1)."""
    test = data.loc[jeu["i_test"]]
    residu = probabilites - jeu["y"][jeu["i_test"]].to_numpy()
    blocs = []
    for colonne in [c for c in jeu["variables"] if c.startswith("genre_")]:
        masque = test[colonne].to_numpy() == 1
        if masque.sum() < minimum:
            continue
        blocs.append(pd.DataFrame({"type de film": colonne[6:], "résidu": residu[masque]}))
    return pd.concat(blocs, ignore_index=True)