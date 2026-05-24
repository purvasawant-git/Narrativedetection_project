"""
topic_model.py  —  Phase 5: BERTopic Topic Modeling  (Per-Week Models)

Fits a SEPARATE BERTopic model for each week's articles.
Extracts topic distributions and labels for each week.
Saves topic_distributions.parquet for Phase 6 (shift detection).

CRITICAL DESIGN: Fit one model PER WEEK, not one global model.
This allows topics to evolve independently week-to-week.
The Jensen-Shannon Divergence in Phase 6 measures how different
the topic mix is between consecutive weeks.

INPUT  : {week: [texts...]} from preprocessing.py (Phase 4)
OUTPUT : data/topic_distributions.parquet + data/bertopic_model/ (saved model)
"""

# ── Windows DLL fix: import torch FIRST so it registers its CUDA lib directory
# before numpy/tokenizers/other native extensions alter the DLL search path.
# Fixes: OSError: [WinError 1114] c10.dll initialization failed.
import os, sys, importlib
_torch_spec = importlib.util.find_spec("torch")
if _torch_spec and _torch_spec.origin:
    _torch_lib = os.path.join(os.path.dirname(_torch_spec.origin), "lib")
    if os.path.isdir(_torch_lib):
        os.add_dll_directory(os.path.abspath(_torch_lib))
import torch  # noqa: E402  — must be before ALL other native imports

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = DATA_DIR / "bertopic_model"
TOPIC_DIST_PATH = DATA_DIR / "topic_distributions.parquet"

DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# BERTopic configuration
# ─────────────────────────────────────────────────────────────────────────────

def build_bertopic_model(n_docs: int = 100) -> BERTopic:
    """
    Build a configured BERTopic instance (unfitted).
    
    Uses:
    - Sentence transformer: all-MiniLM-L6-v2 (fast, ~80MB, good for news)
    - min_topic_size = 3 (small min because some weeks have only 5-10 articles)
    - CountVectorizer with bigrams for keyword extraction
    - UMAP with dynamic neighbors & init to prevent scipy errors on small datasets
    
    Returns
    -------
    BERTopic
        Unfitted model ready to call fit_transform(docs)
    """
    logger.debug("Building BERTopic model configuration...")
    
    # Sentence transformer embedding model
    # Using cache to avoid re-downloading on each week
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # CountVectorizer for topic label extraction
    # stop_words here is ONLY for the keyword extraction, not for embeddings
    vectorizer_model = CountVectorizer(
        ngram_range=(1, 2),  # unigrams + bigrams
        min_df=1,            # word must appear in at least 1 doc (to avoid ValueError when only 1 topic is found)
        stop_words="english"
    )
    
    from umap import UMAP
    # Dynmically adjust UMAP to prevent spectral layout errors on small data
    n_neighbors = min(15, max(2, n_docs - 1))
    umap_init = "random" if n_docs < 15 else "spectral"
    umap_model = UMAP(n_neighbors=n_neighbors, n_components=2, min_dist=0.0, metric='cosine', init=umap_init)
    
    # BERTopic model
    model = BERTopic(
        embedding_model=embedding_model,
        vectorizer_model=vectorizer_model,
        umap_model=umap_model,
        min_topic_size=3,
        nr_topics=None,      # Do not auto-reduce to avoid IndexError when all docs are outliers
        language="english",
        calculate_probabilities=True,  # get soft topic assignments
        verbose=False
    )
    
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Per-week topic modeling
# ─────────────────────────────────────────────────────────────────────────────

def fit_weekly_topics(weekly_docs: Dict[str, List[str]]) -> pd.DataFrame:
    """
    Fit a SEPARATE BERTopic model for each week.
    Extract topic distributions and labels.
    
    This is the core design decision: fitting per-week allows topics
    to evolve independently. The union of topic_ids becomes your
    vocabulary for JSD computation in Phase 6.
    
    Parameters
    ----------
    weekly_docs : dict
        { "2022-01-03": [text1, text2, ...], ... }
        From preprocessing.get_weekly_documents()
    
    Returns
    -------
    pd.DataFrame
        Columns: week_start, topic_id, topic_label, weight, article_count, low_data_week
        One row per (week, topic) pair showing the weight/fraction for that topic that week
    """
    logger.info("=" * 80)
    logger.info("  Phase 5: BERTopic Per-Week Modeling")
    logger.info("=" * 80)
    
    weeks_sorted = sorted(weekly_docs.keys())
    logger.info("Fitting %d weekly models...\n", len(weeks_sorted))
    
    all_rows = []
    
    for week_idx, week_str in enumerate(weeks_sorted, 1):
        docs = weekly_docs[week_str]
        n_articles = len(docs)
        
        # Flag low-data weeks
        low_data = n_articles < 5
        
        # Fit model for this week
        model = build_bertopic_model(n_docs=n_articles)
        topics, probs = model.fit_transform(docs)
        
        # Get topic info
        topic_info = model.get_topic_info()
        n_topics = len([t for t in topic_info["Topic"] if t != -1])
        
        # Compute weights: fraction of articles per topic
        unique_topics = np.unique(topics)
        
        for topic_id in unique_topics:
            count = np.sum(topics == topic_id)
            weight = count / n_articles
            
            # Get topic label (top 3 words)
            if topic_id == -1:
                # Outlier topic
                topic_label = "outlier_uncategorized"
            else:
                try:
                    words = model.get_topic(topic_id)
                    # words is list of (word, freq) tuples
                    top_3_words = [w[0] for w in words[:3]]
                    topic_label = "_".join(top_3_words)
                except Exception:
                    topic_label = f"topic_{topic_id}"
            
            all_rows.append({
                "week_start": week_str,
                "topic_id": int(topic_id),
                "topic_label": topic_label,
                "weight": weight,
                "article_count": n_articles,
                "low_data_week": low_data,
            })
        
        # Log progress
        dominant_topic = max(
            [r for r in all_rows if r["week_start"] == week_str],
            key=lambda x: x["weight"]
        )
        
        logger.info(
            "[%d/%d] Week %s: %d articles → %d topics. "
            "Dominant: %s (%.2f) %s",
            week_idx, len(weeks_sorted), week_str, n_articles, n_topics,
            dominant_topic["topic_label"], dominant_topic["weight"],
            "[low data]" if low_data else ""
        )
    
    df = pd.DataFrame(all_rows)
    
    logger.info("\n" + "=" * 80)
    logger.info("Weekly modeling complete.")
    logger.info("  Total (week, topic) pairs: %d", len(df))
    logger.info("  Unique topics across all weeks: %d", df["topic_id"].nunique())
    logger.info("=" * 80 + "\n")
    
    return df


def get_dominant_topic_per_week(topic_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract the dominant (highest-weight) topic for each week.
    Excludes topic_id = -1 (outlier).
    
    Parameters
    ----------
    topic_df : pd.DataFrame
        Output from fit_weekly_topics()
    
    Returns
    -------
    pd.DataFrame
        Columns: week_start, dominant_topic_id, dominant_topic_label, weight
        One row per week, sorted chronologically
    """
    # Exclude outliers (-1)
    non_outlier = topic_df[topic_df["topic_id"] != -1]
    
    dominant = non_outlier.loc[non_outlier.groupby("week_start")["weight"].idxmax()]
    dominant = dominant[["week_start", "topic_id", "topic_label", "weight"]].copy()
    dominant.columns = ["week_start", "dominant_topic_id", "dominant_topic_label", "weight"]
    
    dominant = dominant.sort_values("week_start").reset_index(drop=True)
    
    # Print quarterly summary
    logger.info("\nQuarterly Topic Summary:")
    
    dominant["_quarter"] = pd.to_datetime(dominant["week_start"]).dt.to_period("Q")
    
    for quarter, group in dominant.groupby("_quarter"):
        # Most common topic in this quarter
        topic_counts = group["dominant_topic_label"].value_counts()
        most_common = topic_counts.index[0]
        
        logger.info("  %s: %s", quarter, most_common)
    
    return dominant.drop(columns=["_quarter"])


def save_topic_model(model: BERTopic, path: str = str(MODEL_DIR)) -> None:
    """
    Save the last fitted BERTopic model.
    
    Note: With per-week models, we save the last week's model.
    For full reproducibility, you'd save all weekly models, but that's
    memory-intensive. The topic_distributions.parquet is the key output.
    
    Parameters
    ----------
    model : BERTopic
        Fitted model
    path : str
        Directory to save to
    """
    try:
        model.save(path)
        logger.info("Saved BERTopic model: %s", path)
    except Exception as exc:
        logger.error("Failed to save model: %s", exc)


def run_full_pipeline(weekly_docs: Dict[str, List[str]],
                      save_path: str = str(TOPIC_DIST_PATH)) -> pd.DataFrame:
    """
    Full Phase 5 pipeline: fit weekly models → extract distributions → save.
    
    Parameters
    ----------
    weekly_docs : dict
        { "2022-01-03": [texts...], ... } from preprocessing.py
    save_path : str
        Where to save topic_distributions.parquet
    
    Returns
    -------
    pd.DataFrame
        Topic distribution matrix, ready for Phase 6
    """
    if not weekly_docs:
        logger.error("Empty weekly_docs provided")
        return pd.DataFrame()
    
    # Fit models
    topic_df = fit_weekly_topics(weekly_docs)
    
    if topic_df.empty:
        logger.error("Topic fitting produced empty dataframe")
        return pd.DataFrame()
    
    # Get dominant topics per week
    dominant = get_dominant_topic_per_week(topic_df)
    
    # Save to parquet (merge with existing if present)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    if Path(save_path).exists():
        existing = pd.read_parquet(save_path, engine="pyarrow")
        topic_df = pd.concat([existing, topic_df], ignore_index=True)
        topic_df.drop_duplicates(subset=["week_start", "topic_id"], keep="last", inplace=True)
        logger.info("Merged with existing topic_distributions.parquet")
    topic_df.to_parquet(save_path, engine="pyarrow", index=False)
    logger.info("\nSaved topic_distributions.parquet: %s", save_path)
    
    return topic_df


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def print_topic_summary(topic_df: pd.DataFrame) -> None:
    """
    Print a summary of topics extracted.
    """
    print("\n" + "=" * 80)
    print("  Topic Distribution Summary")
    print("=" * 80)
    
    print(f"\nTotal (week, topic) pairs: {len(topic_df)}")
    print(f"Unique weeks: {topic_df['week_start'].nunique()}")
    print(f"Unique topics: {topic_df['topic_id'].nunique()}")
    
    print("\nTop topic labels by total weight:")
    top_topics = topic_df.groupby("topic_label")["weight"].sum().sort_values(ascending=False)
    for label, total_weight in top_topics.head(10).items():
        print(f"  {label}: {total_weight:.2f}")
    
    print("\n" + "=" * 80 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI (for testing)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    
    # Import preprocessing to load data
    from preprocessing import run_preprocessing_pipeline
    
    # Run preprocessing first
    weekly_docs = run_preprocessing_pipeline()
    
    if not weekly_docs:
        logger.error("Preprocessing failed")
        exit(1)
    
    # Run full pipeline
    topic_df = run_full_pipeline(weekly_docs)
    
    if not topic_df.empty:
        print_topic_summary(topic_df)
        logger.info("✓ Phase 5 complete. Ready for Phase 6 (Shift Detection)")
    else:
        logger.error("Phase 5 failed")
