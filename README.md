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
4/ dans le dossier notebooks, lancer la commande suivante : `streamlit run dashboard_recommandation.py`

## Utilisation Succès au box-office

Aucun téléchargement ni clé API : les tables nettoyées et le modèle entraîné sont dans le dépôt.

1/ (Optionnel) exécutez `notebooks/01_qualite_donnees.ipynb` pour l'audit de qualité des données
2/ exécutez le notebook `notebooks/02_succes_technique.ipynb` (environ 5 minutes, dont 2 pour les valeurs SHAP)
3/ exécutez `notebooks/03_restitution_metier.ipynb` pour la lecture métier, ou lancez le tableau de bord (voir plus bas)

| Fichier | Contenu |
|---|---|
| `notebooks/01_qualite_donnees.ipynb` | audit de qualité des données : manquants, budgets aberrants, inflation, représentativité |
| `notebooks/02_succes_technique.ipynb` | modèle de succès : cible, choix du modèle, réglage, seuil de décision, contrôles, interprétabilité |
| `src/boxoffice/` | le code appelé par le notebook : collecte, nettoyage, variables, modèle, figures |
| `data/processed/` | les deux tables nettoyées lues par le notebook (3 Mo), issues de MovieLens, TMDB et IMDb |
| `notebooks/models/succes_classification.*` | modèle entraîné et sa carte d'identité |

Le notebook est livré déjà exécuté : il se lit sans rien lancer. Le modèle enregistré dans `notebooks/models/` est celui qu'utilise la partie métier.

### Chiffres clés

- 31 464 films présents dans les trois sources, 9 268 exploitables pour le modèle de succès (11 % des 87 585 films MovieLens).
- Cible : le film rapporte-t-il au moins 2,5 fois son budget ? 38,6 % y parviennent.
- Modèle retenu : gradient boosting, AUC 0,749 sur les films sortis à partir de 2018, jamais vus à l'entraînement.
- Au seuil retenu, 72 % des films recommandés sont rentables, contre 35 % dans le catalogue.

## Restitution métier (direction du studio)

Cette partie traduit le modèle en recommandations d'investissement, sans jargon technique. Elle répond à la question : *quel film produire ensuite ?*

| Fichier | Contenu |
|---|---|
| `notebooks/03_restitution_metier.ipynb` | analyse destinée à la direction : budget, genre, saison, saga, zones de décision, fiabilité par genre, simulateur interactif, recommandations |
| `notebooks/dashboard_succes.py` | tableau de bord Streamlit « Quel film produire ensuite ? » |
| `soutenance_intro_DataVisualisation.pptx` | support de l'introduction de la soutenance (question business, sources, pipeline, entonnoir de sélection) |

### Lancer le tableau de bord

Depuis la racine du dépôt, avec l'environnement virtuel activé :

   cd notebooks
   streamlit run dashboard_succes.py

Le tableau de bord charge le modèle enregistré dans `notebooks/models/` et les tables de `data/processed/` : aucune clé API ni téléchargement n'est nécessaire. Il contient :

- un **simulateur de projet** (genre, budget, suite/univers, période de sortie, durée, classification d'âge, type de studio) qui affiche la probabilité de rentabilité et une décision ;
- cinq onglets d'analyse : budget et saga, genres et budgets, calendrier, fiabilité de l'outil, erreurs par type de film ;
- un encart « Ce que cet outil ne fait pas » rappelant les limites.

### Les trois zones de décision

| Zone | Signification | Films rentables observés (test : 983 films récents) |
|---|---|---|
| Recommander | le projet cumule les caractéristiques gagnantes | 72 % |
| Examiner | signaux contradictoires, analyse humaine nécessaire | 40 % |
| Écarter | configuration historiquement perdante | 15 % |

Taux de référence du catalogue : 35 %.

### Principaux enseignements

- Une suite ou un film issu d'un univers existant réussit environ deux fois plus souvent qu'un film isolé.
- Pour un film isolé, la probabilité de rentabilité diminue quand le budget augmente (près de 6 chances sur 10 à 10 M$, moins de 3 sur 10 à 150 M$). Au-delà de 100 M$, un budget ne se justifie que sur une licence installée.
- Le petit budget est plus sûr quel que soit le genre. Animation, romance et mystère résistent le mieux ; action et science-fiction sont les plus exposées.
- Juin et juillet dominent, suivis des fêtes de fin d'année. Septembre est le pire mois (près de 20 points d'écart avec l'été).

### Limites

- Le modèle donne une probabilité, pas une garantie : environ trois recommandations sur dix se trompent.
- Il ignore le scénario, le budget marketing et la concurrence à la date de sortie.
- Il ne couvre que le cinéma dont budgets et recettes sont publics.
- Sa fiabilité a été divisée par deux pendant la fermeture des salles (2020-2021).

### Reconstituer les sources brutes

Facultatif, environ 1 h 30, avec une clé TMDB gratuite à placer dans `.env` (modèle : `.env.example`).

   python -m boxoffice.ingestion.movielens
   python -m boxoffice.ingestion.tmdb
   python -m boxoffice.ingestion.imdb
   python -m boxoffice.cleaning.reference
   python -m boxoffice.cleaning.movies
