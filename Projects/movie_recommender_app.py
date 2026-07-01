import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st


MODEL_PATH = Path(__file__).parent / "models" / "movie_recommender_deploy_bundle.joblib"
MODEL_URL = (
    "https://huggingface.co/ytim517/kaggle_recommender/resolve/main/"
    "movie_recommender_deploy_bundle.joblib"
)
MODEL_CACHE_PATH = Path(__file__).parent / ".model_cache" / "movie_recommender_deploy_bundle.joblib"
DEFAULT_FEATURE_COLUMNS = [
    "movie_mean",
    "user_mean",
    "movie_n_ratings",
    "user_n_ratings",
    "movie_year",
    "genre_avg",
    "bias_pred",
    "svd_pred",
    "ensemble_pred",
    "content_raw",
    "content_pred",
]


def register_sklearn_pickle_compat_modules():
    try:
        import sklearn._loss._loss as sklearn_loss_ext
    except Exception:
        return

    sys.modules.setdefault("_loss", sklearn_loss_ext)


def resolve_model_path():
    if MODEL_PATH.exists():
        return MODEL_PATH

    MODEL_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if MODEL_CACHE_PATH.exists():
        return MODEL_CACHE_PATH

    with st.spinner("Downloading model bundle from Hugging Face..."):
        response = requests.get(MODEL_URL, stream=True, timeout=60)
        response.raise_for_status()

        temp_path = MODEL_CACHE_PATH.with_suffix(".tmp")
        with temp_path.open("wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)
        temp_path.replace(MODEL_CACHE_PATH)

    return MODEL_CACHE_PATH


@st.cache_resource(show_spinner="Loading recommender model...")
def load_bundle():
    register_sklearn_pickle_compat_modules()
    return joblib.load(resolve_model_path())


def as_series(obj):
    if isinstance(obj, pd.Series):
        return obj
    return pd.Series(obj)


def get_required(bundle, key):
    if key not in bundle:
        raise KeyError(f"Missing required model artifact: {key}")
    return bundle[key]


def movie_table(bundle):
    movies = get_required(bundle, "movies").copy()
    if "title" not in movies.columns:
        movies["title"] = movies["movieId"].astype(str)
    if "genres" not in movies.columns:
        movies["genres"] = ""
    return movies.drop_duplicates("movieId").reset_index(drop=True)


def predict_bias(df, bundle):
    global_mean = float(get_required(bundle, "global_mean"))
    movie_bias = as_series(get_required(bundle, "movie_bias"))
    user_bias = as_series(get_required(bundle, "user_bias"))

    pred = (
        global_mean
        + df["movieId"].map(movie_bias).fillna(0.0)
        + df["userId"].map(user_bias).fillna(0.0)
    )
    return pred.clip(0.5, 5.0).to_numpy(dtype=np.float64)


def predict_svd(df, bundle, model_state):
    global_mean = float(get_required(bundle, "global_mean"))
    user_map = get_required(bundle, "user_map")
    item_map = get_required(bundle, "item_map")
    bu, bi, p_mat, q_mat = model_state

    uc = pd.Series(df["userId"]).map(user_map).fillna(-1).astype(np.int64).to_numpy()
    ic = pd.Series(df["movieId"]).map(item_map).fillna(-1).astype(np.int64).to_numpy()

    pred = np.full(len(df), global_mean, dtype=np.float64)
    has_user = uc >= 0
    has_item = ic >= 0
    pred[has_user] += bu[uc[has_user]]
    pred[has_item] += bi[ic[has_item]]

    both = has_user & has_item
    if both.any():
        pred[both] += np.sum(p_mat[uc[both]] * q_mat[ic[both]], axis=1)

    return np.clip(pred, 0.5, 5.0)


def predict_ensemble(df, bundle):
    models = get_required(bundle, "svd_models")
    names = bundle.get("svd_model_names") or list(models.keys())
    weights = np.asarray(get_required(bundle, "ensemble_weights"), dtype=np.float64)

    preds = np.column_stack([predict_svd(df, bundle, models[name]) for name in names])
    return np.clip(preds @ weights, 0.5, 5.0)


def predict_primary_svd(df, bundle):
    for key in ("scratch_svd_model", "primary_svd_model"):
        if key in bundle:
            return predict_svd(df, bundle, bundle[key])

    if all(key in bundle for key in ("bu", "bi", "P", "Q")):
        return predict_svd(df, bundle, (bundle["bu"], bundle["bi"], bundle["P"], bundle["Q"]))

    svd_models = get_required(bundle, "svd_models")
    best_model_name = bundle.get("best_svd_model_name")
    if best_model_name is None:
        best_model_name = next(iter(svd_models))
    return predict_svd(df, bundle, svd_models[best_model_name])


def content_score(df, bundle):
    movie_content = bundle.get("movie_content")
    movie_content_map = bundle.get("movie_content_map")
    user_profiles = bundle.get("user_profiles")

    scores = np.zeros(len(df), dtype=np.float64)
    if movie_content is None or movie_content_map is None or user_profiles is None:
        return scores

    for idx, (user_id, movie_id) in enumerate(zip(df["userId"], df["movieId"])):
        if user_id in user_profiles and movie_id in movie_content_map:
            scores[idx] = float(
                np.dot(user_profiles[user_id], movie_content[movie_content_map[movie_id]])
            )
    return scores


def predict_content(df, bundle, raw_scores):
    calibrator = bundle.get("calibrator")
    if calibrator is None:
        return np.full(len(df), float(get_required(bundle, "global_mean")), dtype=np.float64)

    pred = calibrator.predict(raw_scores.reshape(-1, 1))
    return np.clip(pred, 0.5, 5.0)


def build_stack_features(df, bundle):
    global_mean = float(get_required(bundle, "global_mean"))
    movie_mean = as_series(get_required(bundle, "movie_mean_map"))
    user_mean = as_series(get_required(bundle, "user_mean_tr"))
    movie_n = as_series(get_required(bundle, "movie_n_ratings_map"))
    user_n = as_series(get_required(bundle, "user_n_ratings_map"))
    movie_year = as_series(get_required(bundle, "movie_year_map"))
    genre_avg = as_series(get_required(bundle, "movie_genre_avg"))
    year_fill = float(bundle.get("year_fill", movie_year.median()))

    bias_pred = predict_bias(df, bundle)
    ensemble_pred = predict_ensemble(df, bundle)
    content_raw = content_score(df, bundle)
    content_pred = predict_content(df, bundle, content_raw)

    feats = pd.DataFrame(index=df.index)
    feats["movie_mean"] = df["movieId"].map(movie_mean).fillna(global_mean).to_numpy()
    feats["user_mean"] = df["userId"].map(user_mean).fillna(global_mean).to_numpy()
    feats["movie_n_ratings"] = np.log1p(df["movieId"].map(movie_n).fillna(0)).to_numpy()
    feats["user_n_ratings"] = np.log1p(df["userId"].map(user_n).fillna(0)).to_numpy()
    feats["movie_year"] = df["movieId"].map(movie_year).fillna(year_fill).to_numpy()
    feats["genre_avg"] = df["movieId"].map(genre_avg).fillna(global_mean).to_numpy()
    feats["bias_pred"] = bias_pred
    feats["svd_pred"] = predict_primary_svd(df, bundle)
    feats["ensemble_pred"] = ensemble_pred
    feats["content_raw"] = content_raw
    feats["content_pred"] = content_pred

    columns = bundle.get("stacker_feature_columns") or DEFAULT_FEATURE_COLUMNS
    return feats.reindex(columns=columns, fill_value=0.0)


def predict_rating(df, bundle):
    stacker = get_required(bundle, "stacker")
    features = build_stack_features(df, bundle)
    pred = stacker.predict(features)
    return np.clip(pred, 0.5, 5.0)


def candidate_movies(movies, bundle, pool_size):
    movie_n = as_series(get_required(bundle, "movie_n_ratings_map"))
    candidates = movies[movies["movieId"].isin(movie_n.index)].copy()
    candidates["rating_count"] = candidates["movieId"].map(movie_n).fillna(0).astype(int)
    return candidates.sort_values("rating_count", ascending=False).head(pool_size)


def recommend_for_user(user_id, bundle, movies, pool_size, top_n):
    candidates = candidate_movies(movies, bundle, pool_size)
    scoring_df = pd.DataFrame(
        {
            "userId": np.full(len(candidates), int(user_id), dtype=np.int64),
            "movieId": candidates["movieId"].to_numpy(dtype=np.int64),
        }
    )
    candidates = candidates.copy()
    candidates["predicted_rating"] = predict_rating(scoring_df, bundle)
    return candidates.sort_values("predicted_rating", ascending=False).head(top_n)


def score_one(user_id, movie_id, bundle):
    df = pd.DataFrame({"userId": [int(user_id)], "movieId": [int(movie_id)]})
    return float(predict_rating(df, bundle)[0])


st.set_page_config(page_title="Movie Recommender", page_icon="film", layout="wide")

st.title("Movie Recommender")

try:
    bundle = load_bundle()
except Exception as exc:
    st.error(f"Could not load model bundle: {exc}")
    st.stop()

movies = movie_table(bundle)
metadata = bundle.get("metadata", {})

with st.sidebar:
    st.header("Model")
    st.caption(f"Bundle: {resolve_model_path().name}")
    st.caption("Source: Hugging Face fallback" if not MODEL_PATH.exists() else "Source: local file")
    if metadata:
        st.json(metadata, expanded=False)

    pool_size = st.slider("Candidate pool", 100, 5000, 1000, step=100)
    top_n = st.slider("Recommendations", 5, 30, 10)

tab_recs, tab_score, tab_movies = st.tabs(["Recommend", "Score Pair", "Movie Search"])

with tab_recs:
    st.subheader("Recommend movies for an existing user")
    user_input = st.text_input("User ID", value="")

    if st.button("Generate Recommendations", type="primary"):
        if not user_input.strip().isdigit():
            st.warning("Enter a numeric user ID.")
        else:
            recs = recommend_for_user(int(user_input), bundle, movies, pool_size, top_n)
            st.dataframe(
                recs[["movieId", "title", "genres", "predicted_rating"]].reset_index(drop=True),
                use_container_width=True,
                hide_index=True,
            )

with tab_score:
    st.subheader("Predict one user/movie rating")
    col_user, col_movie = st.columns(2)
    with col_user:
        score_user = st.text_input("User ID ", value="", key="score_user")
    with col_movie:
        selected_title = st.selectbox(
            "Movie",
            movies["title"].astype(str).tolist(),
            index=0,
        )
        selected_movie_id = int(movies.loc[movies["title"].astype(str) == selected_title, "movieId"].iloc[0])

    if st.button("Predict Rating"):
        if not score_user.strip().isdigit():
            st.warning("Enter a numeric user ID.")
        else:
            rating = score_one(int(score_user), selected_movie_id, bundle)
            st.metric("Predicted rating", f"{rating:.3f}")

with tab_movies:
    st.subheader("Search available movies")
    query = st.text_input("Search title")
    filtered = movies
    if query:
        filtered = movies[movies["title"].astype(str).str.contains(query, case=False, na=False)]

    st.dataframe(
        filtered[["movieId", "title", "genres"]].head(200),
        use_container_width=True,
        hide_index=True,
    )
