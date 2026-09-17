# Movie_Recommendation
# 🎬 Movie Recommender

A content-based movie recommendation system built on the [TMDB 5000 Movies dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata). Pick a movie you like, and it recommends similar ones based on genre, keywords, cast, director, and plot overview — with a Streamlit web app on top.

## How it works

Each movie is turned into a "tag" of words pulled from its genres, keywords, top cast, director, and overview. Genres and keywords are weighted more heavily since they're the strongest signal of what a movie is actually like. These tags are vectorized with **TF-IDF** (unigrams + bigrams) and compared using **cosine similarity** — so "similar" means "shares the most distinctive genre/plot/cast vocabulary," not just "same actor."

Similarity is computed **on the fly** per query from the stored sparse vectors, rather than precomputing and storing a full NxN similarity matrix. This keeps the model artifacts under ~2MB (a precomputed matrix for this dataset would be ~90MB, which also breaks GitHub's 100MB file limit) — and it's still fast, on the order of milliseconds per recommendation.

## Project structure

```
.
├── app.py                    # Streamlit app
├── build_artifacts.py        # Data pipeline: raw CSVs → model artifacts
├── requirements.txt
├── artifacts/
│   ├── movies.pkl            # Movie metadata (title, genres, cast, etc.)
│   └── vectors.npz           # Sparse TF-IDF vectors
├── tmdb_5000_movies.csv      # Raw dataset (not included — see below)
└── tmdb_5000_credits.csv     # Raw dataset (not included — see below)
```

## Setup

**1. Clone the repo and install dependencies**

```bash
git clone <your-repo-url>
cd <your-repo-name>
pip install -r requirements.txt
```

**2. Get the dataset**

Download `tmdb_5000_movies.csv` and `tmdb_5000_credits.csv` from [Kaggle](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) and place them in the project root.

**3. Build the model artifacts**

```bash
python build_artifacts.py
```

This reads the two CSVs and writes `artifacts/movies.pkl` and `artifacts/vectors.npz`. (Skip this step if `artifacts/` is already included in the repo.)

**4. Run the app**

```bash
streamlit run app.py
```

Open the local URL Streamlit prints (usually `http://localhost:8501`).


## Notes for contributors

- Don't commit a virtual environment (`venv/`, `myenv/`, etc.) — see `.gitignore`.
- Don't commit `artifacts/vectors.npz` / `movies.pkl` if you change the pipeline — regenerate them with `build_artifacts.py` instead, so the model always matches the code that built it.

## Possible improvements

- Swap the similarity model for embeddings (e.g. sentence-transformers) for semantic rather than keyword-based matching
- Add a hybrid signal using `vote_average` / `vote_count` to break ties toward better-reviewed movies
- Deploy on Streamlit Community Cloud
