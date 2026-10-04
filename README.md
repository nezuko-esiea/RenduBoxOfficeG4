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

## Contenu du dépôt

| Fichier | Contenu |
|---|---|
| `notebooks/main.ipynb` | **notebook principal** : le modèle de succès (partie A) puis la recommandation (partie B) |
| `notebooks/01_qualite_donnees.ipynb` | audit de qualité des données, livré déjà exécuté |
| `notebooks/02_succes_technique.ipynb` | le modèle de succès seul, repris dans la partie A du notebook principal |
| `notebooks/movielens_clustering_recommender.ipynb` | la recommandation seule, reprise dans la partie B du notebook principal |
| `notebooks/03_restitution_metier.ipynb` | la restitution métier, destinée à la direction du studio |
| `notebooks/dashboard_succes.py` | tableau de bord Streamlit « Quel film produire ensuite ? » |
| `notebooks/dashboard_recommandation.py`, `notebooks/test_model.py` | tableau de bord Streamlit des recommandations, et son script de test en ligne de commande |
| `notebooks/models/` | les modèles entraînés : succès (`succes_classification.*`) et recommandation (`clustering_recommender.pkl`) |
| `src/boxoffice/` | le code appelé par les notebooks du modèle de succès : collecte, nettoyage, variables, modèle, figures |
| `data/processed/` | les cinq tables lues par les notebooks (74 Mo), issues de MovieLens, TMDB et IMDb |
| `soutenance_intro_DataVisualisation.pptx` | support de l'introduction de la soutenance |

## Notebook principal

Aucun téléchargement ni clé API : toutes les tables lues sont dans le dépôt.

1/ ouvrez `notebooks/main.ipynb` et choisissez l'environnement `.venv` comme noyau
2/ exécutez toutes les cellules, de haut en bas

Compter environ 10 minutes, dont 3 à 4 pour les valeurs SHAP et 2 à 3 pour le choix du nombre de clusters, et 7 Go de mémoire vive disponible : la partie B charge les 32 millions de notes MovieLens. Elle réenregistre aussi le modèle de recommandation dans `notebooks/models/`.

Le notebook est livré déjà exécuté : il se lit sans rien lancer.

Le notebook de qualité des données, `notebooks/01_qualite_donnees.ipynb`, n'est pas repris dans le notebook principal : il lit les sources brutes et les tables intermédiaires du pipeline, qui ne sont pas dans le dépôt (environ 2 Go). Il est lui aussi livré déjà exécuté, et ne se rejoue qu'après avoir reconstitué les sources brutes (voir plus bas).

## Utilisation Recommandation

Les notes MovieLens sont dans `data/processed/ratings.parquet` et le modèle entraîné dans `notebooks/models/` : il n'y a rien à télécharger.

1/ exécutez la partie B de `notebooks/main.ipynb`, ou le notebook `movielens_clustering_recommender` seul
2/ dans le dossier notebooks, lancer la commande suivante : `streamlit run dashboard_recommandation.py`
3/ pour tester le modèle en ligne de commande, toujours dans le dossier notebooks : `python test_model.py --user-id 1`

Les affiches des films sont facultatives : elles demandent une clé TMDB gratuite, à placer dans `.env` (modèle : `.env.example`) ou à saisir dans le tableau de bord.

## Utilisation Succès au box-office

1/ exécutez la partie A de `notebooks/main.ipynb`, ou le notebook `notebooks/02_succes_technique.ipynb` seul (environ 5 minutes, dont 2 pour les valeurs SHAP)
2/ pour la lecture métier, exécutez `notebooks/03_restitution_metier.ipynb` ou lancez le tableau de bord (voir plus bas)

Le modèle enregistré dans `notebooks/models/` est celui qu'utilise la partie métier.

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

## Reconstituer les sources brutes

Facultatif, environ 1 h 30, avec une clé TMDB gratuite à placer dans `.env` (modèle : `.env.example`). Nécessaire seulement pour rejouer le notebook de qualité des données.

   python -m boxoffice.ingestion.movielens
   python -m boxoffice.ingestion.tmdb
   python -m boxoffice.ingestion.imdb
   python -m boxoffice.cleaning.reference
   python -m boxoffice.cleaning.movies
