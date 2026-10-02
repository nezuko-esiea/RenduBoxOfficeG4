# Recommandation de films & Succès au box-office

## Objectif
Analyse prédictive et système de recommandation pour guider les investissements de production de films.

- **Données** : MovieLens, API TMDB
- **ML** : Filtrage collaboratif & modèles de succès (régression/classification)

## Installation 

1/ Cloner le dépôt et s'y placer :

   git clone https://github.com/nezuko-esiea/RenduBoxOfficeG4
   cd movie-boxoffice-recommender

2/ Créer un environnement virtuel :
   
   python -m venv .venv
   .venv\Scripts\activate          # Windows
   source .venv/bin/activate       # macOS / Linux

3/ Installer les dépendances :
   pip install -r requirements.txt

## Utilisation Recommandation

1/ télécharger le zip ml-32.zip sur le site https://grouplens.org/datasets/movielens/
2/ décompressez le zip vers le dossier "movie-boxoffice-recommender"
3/ executez le notebook movielens_clustering_recommender
4/ dans le dossier notebooks, lancer la commande suivante streamlit run dashboard_recommandation.py