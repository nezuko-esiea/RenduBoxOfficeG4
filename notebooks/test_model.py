"""
    python test_model.py --user-id 1
    python test_model.py --user-id 1 --top-n 15 --show-history
"""

import argparse
import os
import pickle

import pandas as pd

DATA_DIR = os.path.join("..", "data", "processed")
MODEL_PATH = os.path.join("models", "clustering_recommender.pkl")


def load_model(path=MODEL_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Modèle introuvable : {path}. "
            "Lance d'abord la section 11 de movielens_clustering_recommender.ipynb."
        )
    with open(path, "rb") as f:
        return pickle.load(f)


def load_data(data_dir=DATA_DIR):
    """Charge les notes et les films depuis les .parquet de data/processed.

    Le fichier movies.parquet du dossier contient déjà un schéma enrichi
    (box-office, TMDB/IMDb) sans la colonne "genres" nécessaire à la
    recommandation — c'est pourquoi les films MovieLens bruts (movieId,
    title, genres) sont stockés séparément dans movielens_movies.parquet.
    """
    ratings = pd.read_parquet(
        os.path.join(data_dir, "ratings.parquet"), columns=["userId", "movieId", "rating"]
    )
    movies = pd.read_parquet(os.path.join(data_dir, "movielens_movies.parquet"))
    return ratings, movies


def get_user_cluster(model, user_id):
    user_clusters = model["user_clusters"]
    row = user_clusters.loc[user_clusters["userId"] == user_id, "cluster"]
    return None if row.empty else row.iloc[0]


def popularity_fallback(ratings, movies, user_id, top_n, min_votes):
    """Recommandation de secours (popularité globale) pour un utilisateur
    absent du modèle (cold start)."""
    already_rated = set(ratings.loc[ratings["userId"] == user_id, "movieId"])

    stats = ratings.groupby("movieId")["rating"].agg(mean_rating="mean", n_votes="count").reset_index()
    stats = stats[stats["n_votes"] >= min_votes]

    c = ratings["rating"].mean()
    m = min_votes
    stats["weighted_rating"] = (
        (stats["n_votes"] / (stats["n_votes"] + m)) * stats["mean_rating"]
        + (m / (stats["n_votes"] + m)) * c
    )

    stats = stats[~stats["movieId"].isin(already_rated)]
    top = stats.sort_values("weighted_rating", ascending=False).head(top_n)
    return top.merge(movies[["movieId", "title", "genres"]], on="movieId")[
        ["movieId", "title", "genres", "weighted_rating", "n_votes"]
    ]


def get_user_genre_set(ratings, movies, user_id, min_rating=3.5):
    """Genres des films que l'utilisateur a bien notés (>= min_rating)."""
    user_ratings = ratings[ratings["userId"] == user_id].merge(
        movies[["movieId", "genres"]], on="movieId"
    )
    liked = user_ratings[user_ratings["rating"] >= min_rating]
    genre_set = set()
    for genre_list in liked["genres"].str.split("|"):
        genre_set.update(genre_list)
    return genre_set


def recommend(model, ratings, movies, user_id, top_n=10, min_votes=20, genre_penalty=0.6):
    cluster_id = get_user_cluster(model, user_id)

    if cluster_id is None:
        print(f"[!] Utilisateur {user_id} inconnu du modèle -> repli sur la popularité globale.")
        return popularity_fallback(ratings, movies, user_id, top_n, min_votes)

    already_rated = set(ratings.loc[ratings["userId"] == user_id, "movieId"])
    user_genres = get_user_genre_set(ratings, movies, user_id)
    stats = model["cluster_movie_stats"]

    candidates = stats[(stats["cluster"] == cluster_id) & (~stats["movieId"].isin(already_rated))].copy()

    def has_genre_overlap(genres_str):
        return bool(set(genres_str.split("|")) & user_genres)

    candidates["genre_match"] = candidates["genres"].apply(has_genre_overlap)
    candidates["final_score"] = candidates["weighted_rating"] * candidates["genre_match"].map(
        {True: 1.0, False: genre_penalty}
    )

    top = candidates.sort_values("final_score", ascending=False).head(top_n)
    return top[["movieId", "title", "genres", "weighted_rating", "final_score", "n_votes"]]


def show_user_history(ratings, movies, user_id, top_n=10):
    user_ratings = ratings[ratings["userId"] == user_id].merge(movies, on="movieId")
    if user_ratings.empty:
        print(f"Aucun historique trouvé pour l'utilisateur {user_id}.")
        return
    top = user_ratings.sort_values("rating", ascending=False).head(top_n)
    print(f"\nFilms les mieux notés par l'utilisateur {user_id} (historique) :")
    print(top[["title", "genres", "rating"]].to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="Teste le modèle de recommandation par clustering.")
    parser.add_argument("--user-id", type=int, required=True, help="Identifiant utilisateur MovieLens")
    parser.add_argument("--top-n", type=int, default=10, help="Nombre de recommandations à afficher")
    parser.add_argument("--min-votes", type=int, default=20, help="Seuil min. de notes pour le fallback popularité")
    parser.add_argument(
        "--genre-penalty",
        type=float,
        default=0.6,
        help="Facteur appliqué au score des films sans genre commun avec l'historique (1.0 = pas de pénalité)",
    )
    parser.add_argument("--show-history", action="store_true", help="Affiche aussi les films préférés de l'utilisateur")
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--data-dir", default=DATA_DIR)
    args = parser.parse_args()

    model = load_model(args.model_path)
    ratings, movies = load_data(args.data_dir)

    cluster_id = get_user_cluster(model, args.user_id)
    if cluster_id is not None:
        print(f"Utilisateur {args.user_id} -> cluster {cluster_id}")

    if args.show_history:
        show_user_history(ratings, movies, args.user_id, top_n=args.top_n)

    recs = recommend(
        model,
        ratings,
        movies,
        args.user_id,
        top_n=args.top_n,
        min_votes=args.min_votes,
        genre_penalty=args.genre_penalty,
    )
    print(f"\nTop {args.top_n} recommandations pour l'utilisateur {args.user_id} :")
    print(recs.to_string(index=False))


if __name__ == "__main__":
    main()
