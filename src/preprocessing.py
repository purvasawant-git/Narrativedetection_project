"""
preprocessing.py  —  Phase 4: Text Preprocessing  (Cleaning & Relevance Filtering)

Cleans articles from DuckDB: remove nulls, short content, URLs, HTML.
Filters to Jio-relevant content. Groups by week for BERTopic input.

CRITICAL: Do NOT remove stopwords or lemmatize before BERTopic.
BERTopic uses sentence-transformer embeddings expecting natural language.

INPUT  : data/articles.duckdb (from Phase 2)
OUTPUT : In-memory dict {week: [list of clean texts]} → Phase 5 (topic_model.py)
"""

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import duckdb
import pandas as pd

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
DATA_DB_PATH = PROJECT_ROOT / "data" / "articles.duckdb"

# ─────────────────────────────────────────────────────────────────────────────
# Company terms for relevance filtering
# ─────────────────────────────────────────────────────────────────────────────

COMPANY_TERMS = [
    "jio",
    "reliance industries",
    "mukesh ambani",
    "ril",
]

# ─────────────────────────────────────────────────────────────────────────────
# Text cleaning
# ─────────────────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Clean raw article text for NLP:
    - Lowercase
    - Remove URLs (http/https/ftp patterns)
    - Remove HTML tags
    - Remove email addresses
    - Remove special characters (keep periods, commas for sentence structure)
    - Normalize whitespace
    
    Parameters
    ----------
    text : str or None
        Raw article text
    
    Returns
    -------
    str
        Cleaned text. Empty string if input is None.
    """
    if text is None or not isinstance(text, str):
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Remove URLs (http, https, ftp)
    text = re.sub(r'https?://[^\s]+|ftp://[^\s]+', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove email addresses
    text = re.sub(r'[\w\.-]+@[\w\.-]+\.\w+', '', text)
    
    # Remove special characters EXCEPT periods and commas (preserve sentence structure)
    # Keep: a-z, 0-9, spaces, periods, commas
    text = re.sub(r'[^a-z0-9\s.,]', '', text)
    
    # Normalize whitespace (collapse multiple spaces/newlines)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def is_relevant(text: str) -> bool:
    """
    Check if cleaned text mentions at least one company term.
    Filters out articles that matched keyword queries but don't actually discuss Jio.
    
    Parameters
    ----------
    text : str
        Cleaned (lowercased) text
    
    Returns
    -------
    bool
        True if at least one COMPANY_TERMS is found
    """
    if not text:
        return False
    
    return any(term in text for term in COMPANY_TERMS)


def preprocess_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply cleaning pipeline to article dataframe.
    Filters by: null content, word count, relevance.
    Adds: clean_content, week_start columns.
    
    Steps logged at each stage showing rows removed.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input with columns: article_id, title, content, published_at, etc.
    
    Returns
    -------
    pd.DataFrame
        Cleaned dataframe with columns:
        article_id, clean_content, week_start, source, title
    """
    logger.info("=" * 80)
    logger.info("  Preprocessing Pipeline")
    logger.info("=" * 80)
    
    df = df.copy()
    initial_count = len(df)
    
    # ── Step 1: Drop rows where content is null or empty ──────────────────────
    before = len(df)
    df = df[df["content"].notna()]
    df = df[df["content"].astype(str).str.strip() != ""]
    removed = before - len(df)
    logger.info("Step 1: Dropped null/empty content. Removed: %d. Remaining: %d",
               removed, len(df))
    
    # ── Step 2: Drop rows where word_count < 80 ──────────────────────────────
    before = len(df)
    df = df[df["word_count"] >= 80]
    removed = before - len(df)
    logger.info("Step 2: Dropped short articles (word_count < 80). Removed: %d. Remaining: %d",
               removed, len(df))
    
    # ── Step 3: Clean content → new column clean_content ────────────────────
    logger.info("Step 3: Applying text cleaning (URLs, HTML, noise removal)...")
    df["clean_content"] = df["content"].apply(clean_text)
    logger.info("  Text cleaning complete.")
    
    # ── Step 4: Filter to relevant articles ───────────────────────────────────
    before = len(df)
    df["is_relevant"] = df["clean_content"].apply(is_relevant)
    df = df[df["is_relevant"]]
    removed = before - len(df)
    logger.info("Step 4: Filtered to Jio-relevant content. Removed: %d. Remaining: %d",
               removed, len(df))
    
    # ── Step 5: Add week_start (Monday of ISO week from published_at) ────────
    logger.info("Step 5: Adding week_start column (ISO week grouping)...")
    
    if not pd.api.types.is_datetime64_any_dtype(df["published_at"]):
        df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
    
    # Use pd.Grouper to get week starting Monday (ISO convention)
    # But we need to create it directly: subtract days from the date to get to Monday
    df["week_start"] = df["published_at"].dt.to_period("W").dt.start_time
    # Ensure it's Monday (ISO week convention)
    # Period.start_time gives Monday by default for 'W'
    
    logger.info("  Week grouping complete.")
    
    # ── Step 6: Drop duplicate titles (keep first by published_at) ───────────
    before = len(df)
    df = df.sort_values("published_at").drop_duplicates(
        subset=["title"], keep="first"
    )
    removed = before - len(df)
    logger.info("Step 6: Dropped duplicate titles. Removed: %d. Remaining: %d",
               removed, len(df))
    
    # ── Final selection ───────────────────────────────────────────────────────
    df = df[["article_id", "clean_content", "week_start", "source", "title"]].copy()
    
    logger.info("\n" + "=" * 80)
    logger.info("Pipeline Summary:")
    logger.info("  Initial articles: %d", initial_count)
    logger.info("  Final articles: %d", len(df))
    logger.info("  Total removed: %d (%.1f%%)", initial_count - len(df),
               (initial_count - len(df)) / initial_count * 100 if initial_count > 0 else 0)
    logger.info("=" * 80 + "\n")
    
    return df


def get_weekly_documents(df: pd.DataFrame,
                        min_articles: int = 3) -> Dict[str, List[str]]:
    """
    Group cleaned articles by week and create a dict of document lists.
    Excludes weeks with fewer than min_articles articles.
    
    Parameters
    ----------
    df : pd.DataFrame
        Output from preprocess_pipeline()
        Must have columns: week_start, clean_content
    min_articles : int, default 3
        Minimum articles per week to include
    
    Returns
    -------
    dict
        Format: { "2022-01-03": ["article text...", "article text..."], ... }
        Keys are ISO date strings (Monday of that week)
        Values are lists of clean article texts
    """
    if df.empty:
        logger.warning("Empty dataframe passed to get_weekly_documents()")
        return {}
    
    # Group by week
    weekly_docs = {}
    weeks_excluded = 0
    
    for week_start, group in df.groupby("week_start"):
        # Convert to ISO date string (YYYY-MM-DD)
        week_key = pd.Timestamp(week_start).strftime("%Y-%m-%d")
        
        texts = group["clean_content"].tolist()
        
        # Exclude weeks below minimum
        if len(texts) < min_articles:
            weeks_excluded += 1
            logger.debug("Excluding week %s: only %d articles (< %d minimum)",
                        week_key, len(texts), min_articles)
            continue
        
        weekly_docs[week_key] = texts
    
    total_articles = df.shape[0]
    total_weeks = len(weekly_docs)
    avg_articles_per_week = (
        total_articles / total_weeks if total_weeks > 0 else 0
    )
    
    logger.info(
        "Built weekly documents: %d weeks, avg %.1f articles/week, "
        "%d weeks excluded (< %d articles)",
        total_weeks, avg_articles_per_week, weeks_excluded, min_articles
    )
    
    return weekly_docs


# ─────────────────────────────────────────────────────────────────────────────
# Full pipeline orchestration
# ─────────────────────────────────────────────────────────────────────────────

def run_preprocessing_pipeline(db_path: str = str(DATA_DB_PATH),
                               min_articles_per_week: int = 3) -> Dict[str, List[str]]:
    """
    Load articles from DuckDB, preprocess, and return weekly document groups.
    
    This is the pipeline entry point from Phase 3 to Phase 5.
    
    Parameters
    ----------
    db_path : str
        Path to articles.duckdb
    min_articles_per_week : int
        Minimum articles per week to include in output
    
    Returns
    -------
    dict
        { "YYYY-MM-DD": [texts...], ... } ready for BERTopic
    """
    logger.info("Starting Phase 4: Text Preprocessing")
    
    # Load articles from DuckDB
    if not Path(db_path).exists():
        logger.error("Database not found: %s", db_path)
        raise FileNotFoundError(f"Database not found: {db_path}")
    
    conn = duckdb.connect(db_path, read_only=True)
    
    try:
        articles_df = conn.execute(
            "SELECT article_id, title, content, source, published_at, word_count "
            "FROM articles ORDER BY published_at ASC"
        ).df()
        
        logger.info("Loaded %d articles from database", len(articles_df))
    finally:
        conn.close()
    
    if articles_df.empty:
        logger.error("No articles found in database")
        return {}
    
    # Preprocess
    cleaned_df = preprocess_pipeline(articles_df)
    
    if cleaned_df.empty:
        logger.error("Preprocessing produced empty dataframe")
        return {}
    
    # Group by week
    weekly_docs = get_weekly_documents(cleaned_df, min_articles=min_articles_per_week)
    
    return weekly_docs


# ─────────────────────────────────────────────────────────────────────────────
# Utility: inspect preprocessing results
# ─────────────────────────────────────────────────────────────────────────────

def print_sample_documents(weekly_docs: Dict[str, List[str]], n_weeks: int = 3) -> None:
    """
    Print sample documents from first n weeks for quality inspection.
    
    Parameters
    ----------
    weekly_docs : dict
        Output from get_weekly_documents()
    n_weeks : int
        Number of weeks to sample
    """
    weeks = sorted(weekly_docs.keys())[:n_weeks]
    
    print("\n" + "=" * 80)
    print("  Sample Preprocessed Documents")
    print("=" * 80)
    
    for week in weeks:
        texts = weekly_docs[week]
        print(f"\nWeek: {week}  ({len(texts)} articles)")
        print("-" * 80)
        
        # Show first and last text
        if len(texts) > 0:
            print("\n[First article]")
            print(texts[0][:300] + ("..." if len(texts[0]) > 300 else ""))
        
        if len(texts) > 1:
            print("\n[Last article]")
            print(texts[-1][:300] + ("..." if len(texts[-1]) > 300 else ""))
        
        print()
    
    print("=" * 80 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    
    # Run full pipeline
    weekly_docs = run_preprocessing_pipeline(
        db_path=str(DATA_DB_PATH),
        min_articles_per_week=3
    )
    
    if weekly_docs:
        logger.info("\n✓ Preprocessing complete. Ready for Phase 5 (BERTopic modeling)")
        logger.info("  Input to topic_model.py: %d weeks of documents", len(weekly_docs))
        
        # Optionally print samples
        print_sample_documents(weekly_docs, n_weeks=3)
    else:
        logger.error("Preprocessing failed or produced empty result")
