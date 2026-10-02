# Recommandation de films & Succès au box-office

## Objectif
Analyse prédictive et système de recommandation pour guider les investissements de production de films.

- **Données** : MovieLens, API TMDB, IMDb
- **ML** : Filtrage collaboratif & modèles de succès (régression/classification)

## Installation

1/ Cloner le dépôt et s'y placer :

   git clone https://github.com/nezuko-esiea/RenduBoxOfficeG4
   cd RenduBoxOfficeG4

2/ Créer un environnement virtuel :

   python -m venv .venv
   .venv\Scripts\activate          # Windows
   source .venv/bin/activate       # macOS / Linux

3/ Installer les dépendances :
   pip install -r requirements.txt

## Utilisation Recommandation

1/ télécharger le zip ml-32.zip sur le site https://grouplens.org/datasets/movielens/
2/ décompressez le zip vers le dossier du dépôt
3/ executez le notebook movielens_clustering_recommender
4/ dans le dossier notebooks, lancer la commande suivante streamlit run dashboard_recommandation.py

## Utilisation Succès au box-office

Aucun téléchargement ni clé API : les tables nettoyées et le modèle entraîné sont dans le dépôt.

1/ exécutez le notebook `notebooks/02_succes_technique.ipynb` (environ 5 minutes, dont 2 pour les valeurs SHAP)

| Fichier | Contenu |
|---|---|
| `notebooks/02_succes_technique.ipynb` | modèle de succès : cible, choix du modèle, réglage, seuil de décision, contrôles, interprétabilité |
| `src/boxoffice/` | le code appelé par le notebook : collecte, nettoyage, variables, modèle, figures |
| `data/processed/` | les deux tables nettoyées lues par le notebook (3 Mo), issues de MovieLens, TMDB et IMDb |
| `notebooks/models/succes_classification.*` | modèle entraîné et sa carte d'identité |

Le notebook est livré déjà exécuté : il se lit sans rien lancer. Le modèle enregistré dans `notebooks/models/` est celui qu'utilise la partie métier.

### Chiffres clés

- 31 464 films présents dans les trois sources, 9 268 exploitables pour le modèle de succès.
- Cible : le film rapporte-t-il au moins 2,5 fois son budget ? 38,6 % y parviennent.
- Modèle retenu : gradient boosting, AUC 0,749 sur les films sortis à partir de 2018, jamais vus à l'entraînement.
- Au seuil retenu, 72 % des films recommandés sont rentables, contre 35 % dans le catalogue.
### Reconstituer les sources brutes

Facultatif, environ 1 h 30, avec une clé TMDB gratuite à placer dans `.env` (modèle : `.env.example`).

   python -m boxoffice.ingestion.movielens
   python -m boxoffice.ingestion.tmdb
   python -m boxoffice.ingestion.imdb
   python -m boxoffice.cleaning.reference
   python -m boxoffice.cleaning.movies
