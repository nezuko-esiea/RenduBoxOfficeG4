import os

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

from test_model import (
    DATA_DIR,
    MODEL_PATH,
    get_user_cluster,
    load_data,
    load_model,
    recommend,
)

TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w342"
TMDB_API_BASE = "https://api.themoviedb.org/3/movie"

MIN_VOTES_CLUSTER_MOVIES = 1000

st.set_page_config(page_title="MovieLens — Recommandations & Clusters", layout="wide")

@st.cache_resource
def get_model(path=MODEL_PATH):
    return load_model(path)


@st.cache_data
def get_data(data_dir=DATA_DIR):
    return load_data(data_dir)


@st.cache_data
def get_links(data_dir=DATA_DIR):
    path = os.path.join(data_dir, "links.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, dtype={"movieId": "int32"})


@st.cache_data(show_spinner=False)
def fetch_poster_url(tmdb_id, api_key):
    if not api_key or pd.isna(tmdb_id):
        return None
    try:
        resp = requests.get(
            f"{TMDB_API_BASE}/{int(tmdb_id)}",
            params={"api_key": api_key},
            timeout=5,
        )
        resp.raise_for_status()
        poster_path = resp.json().get("poster_path")
        return f"{TMDB_IMAGE_BASE}{poster_path}" if poster_path else None
    except requests.RequestException:
        return None


def attach_posters(df, links, api_key):
    if links is None or not api_key:
        return df.assign(poster_url=None)
    merged = df.merge(links[["movieId", "tmdbId"]], on="movieId", how="left")
    merged["poster_url"] = merged["tmdbId"].apply(lambda t: fetch_poster_url(t, api_key))
    return merged


def show_movie_grid(df, columns=5):
    if df.empty:
        st.info("Aucun film à afficher.")
        return
    score_col = "final_score" if "final_score" in df.columns else "weighted_rating"
    rows = [df.iloc[i : i + columns] for i in range(0, len(df), columns)]
    for row in rows:
        cols = st.columns(columns)
        for col, (_, movie) in zip(cols, row.iterrows()):
            with col:
                if movie.get("poster_url"):
                    st.image(movie["poster_url"], use_container_width=True)
                else:
                    st.markdown(
                        "<div style='height:220px;display:flex;align-items:center;"
                        "justify-content:center;background:#eee;border-radius:8px;"
                        "text-align:center;padding:8px;color:#666;'>🎬<br>Pas d'affiche</div>",
                        unsafe_allow_html=True,
                    )
                st.caption(f"**{movie['title']}**")
                st.caption(movie.get("genres", ""))
                if score_col in movie and pd.notna(movie[score_col]):
                    st.caption(f"Score : {movie[score_col]:.2f}")



GENRE_NICKNAMES = {
    "Action": "Les amateurs d'action",
    "Adventure": "Les aventuriers",
    "Animation": "Les grands enfants",
    "Children": "Les habitués des films en famille",
    "Comedy": "Les rigolos",
    "Crime": "Les enquêteurs",
    "Documentary": "Les curieux du réel",
    "Drama": "Les amateurs de drames",
    "Fantasy": "Les rêveurs",
    "Film-Noir": "Les adorateurs de films noirs",
    "Horror": "Les amateurs de frissons",
    "IMAX": "Les amateurs de grand spectacle",
    "Musical": "Les mélomanes",
    "Mystery": "Les limiers",
    "Romance": "Les romantiques",
    "Sci-Fi": "Les explorateurs de science-fiction",
    "Thriller": "Les amateurs de suspense",
    "War": "Les stratèges de guerre",
    "Western": "Les cow-boys",
    "(no genres listed)": "Les ovnis cinématographiques",
}


def build_cluster_names(genre_profile):
    names = {}
    used = set()
    for cid in genre_profile.index:
        ranked_genres = genre_profile.loc[cid].sort_values(ascending=False).index.tolist()
        name = None
        for genre in ranked_genres:
            candidate = GENRE_NICKNAMES.get(genre, f"Les fans de {genre}")
            if candidate not in used:
                name = candidate
                break
        if name is None:
            name = f"{GENRE_NICKNAMES.get(ranked_genres[0], ranked_genres[0])} (bis)"
        used.add(name)
        names[cid] = name
    return names


# ----------------------------------------------------------------------------
# Chargement des données et du modèle
# ----------------------------------------------------------------------------

st.title("🎬 MovieLens — Recommandations & Clusters")

try:
    model = get_model()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

ratings, movies = get_data()
links = get_links()

genre_profile = model["cluster_genre_profile"]
cluster_stats = model["cluster_movie_stats"]
user_clusters = model["user_clusters"]
cluster_ids = sorted(genre_profile.index.tolist())
cluster_names = build_cluster_names(genre_profile)


def cluster_label(cid):
    return f"{cluster_names[cid]} (cluster {cid})"


with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input(
        "Clé API TMDB (optionnelle, pour les affiches)",
        value=os.environ.get("TMDB_API_KEY", "09c183a9e21038d5bc47c4ae460b9559"),
        type="password",
    )
    if links is None:
        st.caption("links.csv introuvable dans le dossier de données — pas d'affiches possibles.")

tab_user, tab_cluster, tab_compare = st.tabs(
    ["Recommandations par utilisateur", "Profil d'un cluster", "Comparer deux clusters"]
)


with tab_user:
    all_users = sorted(ratings["userId"].unique().tolist())
    user_id = st.selectbox("Utilisateur", all_users, index=0)
    top_n = st.slider("Nombre de recommandations", 5, 30, 10)

    cluster_id = get_user_cluster(model, user_id)
    if cluster_id is not None:
        cluster_size = (user_clusters["cluster"] == cluster_id).sum()
        st.write(
            f"Profil de l'utilisateur **{user_id}** : **{cluster_label(cluster_id)}** "
            f"({cluster_size:,} utilisateurs)"
        )
    else:
        st.write(f"Utilisateur **{user_id}** inconnu du modèle → repli sur la popularité globale.")

    with st.expander("Historique de l'utilisateur (films les mieux notés)"):
        history = ratings[ratings["userId"] == user_id].merge(movies, on="movieId")
        history = history.sort_values("rating", ascending=False).head(15)
        st.dataframe(history[["title", "genres", "rating"]], use_container_width=True, hide_index=True)

    recs = recommend(model, ratings, movies, user_id, top_n=top_n)
    recs = attach_posters(recs, links, api_key)

    st.subheader("Affiches recommandées")
    show_movie_grid(recs)

    st.subheader("Détail")
    st.dataframe(
        recs.drop(columns=["poster_url", "tmdbId"], errors="ignore"),
        use_container_width=True,
        hide_index=True,
    )

with tab_cluster:
    cid = st.selectbox("Cluster", cluster_ids, format_func=cluster_label, key="single_cluster")
    size = (user_clusters["cluster"] == cid).sum()
    st.write(f"**{cluster_names[cid]}** — {size:,} utilisateurs")

    top_genres = genre_profile.loc[cid].sort_values(ascending=False).head(12)
    fig = px.bar(
        top_genres.iloc[::-1],
        orientation="h",
        labels={"value": "Sur-représentation (lift)", "index": "Genre"},
        title=f"Genres sur-représentés — {cluster_names[cid]}",
    )
    fig.add_vline(x=1, line_dash="dash", line_color="gray")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Films les plus distinctifs du cluster")
    top_movies = (
        cluster_stats[
            (cluster_stats["cluster"] == cid)
            & (cluster_stats["n_votes"] >= MIN_VOTES_CLUSTER_MOVIES)
        ]
        .sort_values("distinctive_score", ascending=False)
        .head(15)
    )
    top_movies = attach_posters(top_movies, links, api_key)
    show_movie_grid(top_movies)


with tab_compare:
    col_a, col_b = st.columns(2)
    with col_a:
        cid_a = st.selectbox(
            "Cluster A", cluster_ids, index=0, format_func=cluster_label, key="cluster_a"
        )
    with col_b:
        default_b_index = 1 if len(cluster_ids) > 1 else 0
        cid_b = st.selectbox(
            "Cluster B",
            cluster_ids,
            index=default_b_index,
            format_func=cluster_label,
            key="cluster_b",
        )

    compare_df = genre_profile.loc[[cid_a, cid_b]].T.reset_index()
    compare_df.columns = ["genre", cluster_names[cid_a], cluster_names[cid_b]]
    compare_df = compare_df.sort_values(cluster_names[cid_a], ascending=False)
    compare_long = compare_df.melt(id_vars="genre", var_name="cluster", value_name="lift")

    fig = px.bar(
        compare_long,
        x="genre",
        y="lift",
        color="cluster",
        barmode="group",
        title=f"Comparaison des genres — {cluster_names[cid_a]} vs {cluster_names[cid_b]}",
    )
    fig.add_hline(y=1, line_dash="dash", line_color="gray")
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Films distinctifs côte à côte")
    col_a, col_b = st.columns(2)
    for col, cid in [(col_a, cid_a), (col_b, cid_b)]:
        with col:
            st.markdown(f"**{cluster_names[cid]}**")
            top = (
                cluster_stats[
                    (cluster_stats["cluster"] == cid)
                    & (cluster_stats["n_votes"] >= MIN_VOTES_CLUSTER_MOVIES)
                ]
                .sort_values("distinctive_score", ascending=False)
                .head(8)
            )
            st.dataframe(
                top[["title", "genres", "distinctive_score"]],
                hide_index=True,
                use_container_width=True,
            )