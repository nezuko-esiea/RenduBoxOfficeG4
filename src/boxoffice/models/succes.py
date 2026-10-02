"""Enregistrement et relecture des modèles de succès, avec leurs métadonnées."""
import json
from datetime import datetime, timezone

import joblib
import sklearn

from boxoffice import config

CLASSIFICATION = "succes_classification"
REGRESSION = "succes_regression_roi"


def chemins(nom: str = CLASSIFICATION) -> tuple:
    return config.MODELS / f"{nom}.joblib", config.MODELS / f"{nom}.json"


def sauvegarder(modele, variables, metriques: dict, cible: str = "is_hit", seuil: float | None = None,
                description: str = "", nom: str = CLASSIFICATION) -> dict:
    fichier_modele, fichier_carte = chemins(nom)
    config.MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump(modele, fichier_modele)
    carte = {
        "cible": cible,
        "description": description,
        "modele": type(modele).__name__,
        "hyperparametres": {c: v for c, v in modele.get_params(deep=False).items() if not callable(v)},
        "variables": list(variables),
        "seuil_decision": seuil,
        "metriques": {nom_metrique: round(float(valeur), 4) for nom_metrique, valeur in metriques.items()},
        "entraine_le": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "versions": {"scikit-learn": sklearn.__version__, "joblib": joblib.__version__},
        "fichier": fichier_modele.name,
    }
    fichier_carte.write_text(json.dumps(carte, ensure_ascii=False, indent=2), encoding="utf-8")
    return carte


def charger(nom: str = CLASSIFICATION) -> tuple:
    fichier_modele, fichier_carte = chemins(nom)
    if not fichier_modele.exists():
        raise FileNotFoundError(f"modèle absent : {fichier_modele}")
    return joblib.load(fichier_modele), json.loads(fichier_carte.read_text(encoding="utf-8"))
