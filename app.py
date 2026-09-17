"""
Movie Recommender - Streamlit app
----------------------------------
Loads the precomputed model artifacts (movies.pkl) and lets
the user pick a movie to get similar recommendations.

Run with:
    streamlit run app.py

Expected folder layout (both must sit next to this file, or inside an
"artifacts" subfolder):
    app.py
    artifacts/
        movies.pkl
        vectors.npz

If you don't have these files yet, generate them by running
build_artifacts.py against tmdb_5000_movies.csv / tmdb_5000_credits.csv.

Note: similarity is computed on-the-fly from the sparse TF-IDF vectors
(vectors.npz) rather than loading a precomputed NxN similarity matrix.
That keeps the artifact under ~2MB instead of ~90MB (which is what a dense
similarity matrix costs, and GitHub rejects files over 100MB anyway) - the
per-query cost is a few milliseconds, so there's no real downside.

Optional: set a TMDB API key (as an environment variable TMDB_API_KEY, or
paste it into the sidebar field) to show poster images. Without a key the
app still works, just without poster art.
"""

import os
import pickle
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------------------------------------------------------
# Config / paths
# --------------------------------------------------------------------------
APP_DIR = Path(__file__).parent
ARTIFACT_CANDIDATES = [APP_DIR / "artifacts", APP_DIR]

st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")


def find_artifact(filename: str) -> Path:
    for folder in ARTIFACT_CANDIDATES:
        candidate = folder / filename
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Couldn't find {filename}. Expected it in {APP_DIR / 'artifacts'} or {APP_DIR}."
    )


@st.cache_resource
def load_model():
    movies = pickle.load(open(find_artifact("movies.pkl"), "rb"))
    vectors = sparse.load_npz(find_artifact("vectors.npz"))
    return movies, vectors


# @st.cache_data(show_spinner=False)
# def fetch_poster(movie_title: str, api_key: str):
#     """Best-effort TMDB poster lookup. Returns None on any failure so the
#     app never crashes just because a poster couldn't be found."""
#     if not api_key:
#         return None
#     try:
#         resp = requests.get(
#             "https://api.themoviedb.org/3/search/movie",
#             params={"api_key": api_key, "query": movie_title},
#             timeout=5,
#         )
#         resp.raise_for_status()
#         results = resp.json().get("results", [])
#         if results and results[0].get("poster_path"):
#             return f"https://image.tmdb.org/t/p/w500{results[0]['poster_path']}"
#     except requests.RequestException:
#         pass
#     return None


def recommend(movies: pd.DataFrame, vectors, title: str, top_n: int = 8):
    matches = movies.index[movies["title"] == title]
    if len(matches) == 0:
        return pd.DataFrame()
    idx = matches[0]
    # Compute similarity for just this one movie against the whole catalog,
    # on demand, instead of loading a precomputed NxN matrix.
    sims = cosine_similarity(vectors[idx], vectors).ravel()
    scores = list(enumerate(sims))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1 : top_n + 1]
    rows = []
    for i, score in scores:
        row = movies.iloc[i]
        rows.append(
            {
                "title": row["title"],
                "genres": ", ".join(row.get("genres", [])) if isinstance(row.get("genres"), list) else "",
                "vote_average": row.get("vote_average", None),
                "similarity": round(float(score), 3),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
st.title("🎬 Movie Recommender")
st.caption("Content-based recommendations from genres, keywords, cast, director, and plot overview.")

try:
    movies, vectors = load_model()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

with st.sidebar:
    st.header("Settings")
    top_n = st.slider("Number of recommendations", min_value=3, max_value=15, value=8)
    # api_key = st.text_input(
    #     "TMDB API key (optional, for poster images)",
    #     value=os.environ.get("TMDB_API_KEY", ""),
    #     type="password",
    #     help="Get a free key at https://www.themoviedb.org/settings/api",
    # )
    st.markdown("---")
    st.markdown(f"**Movies in catalog:** {len(movies):,}")

titles = sorted(movies["title"].tolist())
selected = st.selectbox("Pick a movie you like:", titles, index=titles.index("Batman Begins") if "Batman Begins" in titles else 0)

if st.button("Recommend", type="primary") or selected:
    results = recommend(movies, vectors, selected, top_n=top_n)

    if results.empty:
        st.warning(f"No data found for '{selected}'.")
    else:
        st.subheader(f"Because you liked *{selected}*")

        cols = st.columns(4)
        for i, row in results.iterrows():
            col = cols[i % 4]
            with col:
                # poster_url = fetch_poster(row["title"], api_key) if api_key else None
                # if poster_url:
                #     st.image(poster_url, use_container_width=True)
                # else:
                #     st.markdown(
                #         "<div style='height:280px;background:#222;border-radius:8px;"
                #         "display:flex;align-items:center;justify-content:center;color:#888;'>"
                #         "No poster</div>",
                #         unsafe_allow_html=True,
                #     )
                st.markdown(f"**{row['title']}**")
                st.caption(row["genres"])
                st.progress(min(max(row["similarity"], 0.0), 1.0), text=f"similarity {row['similarity']}")

        with st.expander("Show as table"):
            st.dataframe(results, use_container_width=True, hide_index=True)
