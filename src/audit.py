"""
audit.py  —  Phase 3: Data Audit  (Quality & Coverage Check)

Validates the articles database before NLP phases.
Checks: weekly coverage, content completeness, duplicate detection, source breakdown.
Produces: console report + audit_coverage.png chart.

INPUT  : data/articles.duckdb (from Phase 2)
OUTPUT : data/audit_coverage.png + detailed console report
"""

import logging
from pathlib import Path
from difflib import SequenceMatcher
from collections import Counter
from typing import Optional

import duckdb
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

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
PLOTS_DIR = PROJECT_ROOT / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Audit functions
# ─────────────────────────────────────────────────────────────────────────────

def run_audit(conn: duckdb.DuckDBPyConnection) -> dict:
    """
    Comprehensive data quality audit. Returns a summary dictionary with
    coverage analysis, source breakdown, content quality metrics.
    
    Returns
    -------
    dict with keys:
        - total_articles (int)
        - date_range (tuple: min_date, max_date)
        - articles_per_week (pd.DataFrame: week_start, article_count)
        - coverage_gaps (list of ISO week strings with count < 3)
        - source_breakdown (pd.DataFrame: source, count, pct)
        - keyword_breakdown (pd.DataFrame: keyword_group, count)
        - median_word_count (float)
        - pct_content_missing (float)
        - pct_short_content (float, count < 50 words)
        - top_sources (list of top 5 source strings)
    """
    audit_dict = {}
    
    # ── Total articles ────────────────────────────────────────────────────────
    total = conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
    audit_dict["total_articles"] = total
    logger.info("Total articles in database: %d", total)
    
    if total == 0:
        logger.warning("Database is empty — no audit possible.")
        return audit_dict
    
    # ── Date range ────────────────────────────────────────────────────────────
    row = conn.execute(
        "SELECT MIN(published_at), MAX(published_at) FROM articles "
        "WHERE published_at IS NOT NULL"
    ).fetchone()
    
    min_date, max_date = row[0], row[1]
    audit_dict["date_range"] = (min_date, max_date)
    logger.info("Date range: %s → %s", min_date, max_date)
    
    # ── Articles per week (using existing get_weekly_counts logic) ────────────
    weekly_sql = """
        SELECT
            DATE_TRUNC('week', published_at) AS week_start,
            COUNT(*) AS article_count
        FROM articles
        WHERE published_at IS NOT NULL
        GROUP BY week_start
        ORDER BY week_start
    """
    try:
        weekly_df = conn.execute(weekly_sql).df()
        audit_dict["articles_per_week"] = weekly_df
        logger.info("Weekly coverage: %d weeks, avg %.1f articles/week",
                    len(weekly_df),
                    weekly_df["article_count"].mean() if len(weekly_df) > 0 else 0)
    except Exception as exc:
        logger.error("Failed to compute weekly counts: %s", exc)
        audit_dict["articles_per_week"] = pd.DataFrame()
    
    # ── Coverage gaps (weeks with < 3 articles) ───────────────────────────────
    if not audit_dict["articles_per_week"].empty:
        gaps = audit_dict["articles_per_week"][
            audit_dict["articles_per_week"]["article_count"] < 3
        ]
        gap_list = gaps["week_start"].astype(str).tolist()
        audit_dict["coverage_gaps"] = gap_list
        
        if len(gap_list) > 0:
            logger.warning("Coverage gaps: %d weeks with < 3 articles", len(gap_list))
            for w in gap_list[:10]:  # Log first 10
                count = gaps[gaps["week_start"].astype(str) == w][
                    "article_count"
                ].values[0]
                logger.warning("  WARNING: Week %s — only %d article(s)", w, count)
    else:
        audit_dict["coverage_gaps"] = []
    
    # ── Source breakdown ──────────────────────────────────────────────────────
    source_sql = """
        SELECT source, COUNT(*) AS count
        FROM articles
        WHERE source IS NOT NULL AND source != ''
        GROUP BY source
        ORDER BY count DESC
    """
    try:
        source_df = conn.execute(source_sql).df()
        if not source_df.empty:
            source_df["pct"] = (source_df["count"] / total * 100).round(2)
            audit_dict["source_breakdown"] = source_df
            
            top_sources = source_df["source"].head(5).tolist()
            audit_dict["top_sources"] = top_sources
            
            logger.info("Top 5 sources:")
            for idx, row in source_df.head(5).iterrows():
                logger.info("  %s: %d articles (%.1f%%)", row["source"],
                           row["count"], row["pct"])
    except Exception as exc:
        logger.error("Failed to compute source breakdown: %s", exc)
        audit_dict["source_breakdown"] = pd.DataFrame()
        audit_dict["top_sources"] = []
    
    # ── Keyword group breakdown ───────────────────────────────────────────────
    keyword_sql = """
        SELECT keyword_group, COUNT(*) AS count
        FROM articles
        WHERE keyword_group IS NOT NULL AND keyword_group != ''
        GROUP BY keyword_group
        ORDER BY count DESC
    """
    try:
        keyword_df = conn.execute(keyword_sql).df()
        audit_dict["keyword_breakdown"] = keyword_df
        
        logger.info("Top keyword groups:")
        for idx, row in keyword_df.head(10).iterrows():
            logger.info("  %s: %d articles", row["keyword_group"], row["count"])
    except Exception as exc:
        logger.error("Failed to compute keyword breakdown: %s", exc)
        audit_dict["keyword_breakdown"] = pd.DataFrame()
    
    # ── Content quality metrics ───────────────────────────────────────────────
    quality_sql = """
        SELECT
            COUNT(*) FILTER (WHERE content IS NULL OR content = '') AS missing_count,
            COUNT(*) FILTER (WHERE word_count < 50) AS short_count,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY word_count) AS median_words
        FROM articles
    """
    try:
        quality_row = conn.execute(quality_sql).fetchone()
        missing_count, short_count, median_words = quality_row
        
        pct_missing = (missing_count / total * 100) if total > 0 else 0.0
        pct_short = (short_count / total * 100) if total > 0 else 0.0
        
        audit_dict["pct_content_missing"] = pct_missing
        audit_dict["pct_short_content"] = pct_short
        audit_dict["median_word_count"] = float(median_words) if median_words else 0.0
        
        logger.info("Content quality:")
        logger.info("  Missing content: %d articles (%.1f%%)", missing_count, pct_missing)
        logger.info("  Short content (< 50 words): %d articles (%.1f%%)",
                   short_count, pct_short)
        logger.info("  Median word count: %.0f words", median_words or 0)
        
        if pct_missing > 10.0:
            logger.critical("CRITICAL: Over 10%% of content is missing or empty")
        if pct_short > 20.0:
            logger.warning("WARNING: Over 20%% of content is very short (< 50 words)")
    except Exception as exc:
        logger.error("Failed to compute content quality: %s", exc)
        audit_dict["pct_content_missing"] = 0.0
        audit_dict["pct_short_content"] = 0.0
        audit_dict["median_word_count"] = 0.0
    
    return audit_dict


def print_audit_report(audit_dict: dict) -> None:
    """
    Print a clean, readable audit summary to console.
    Flags critical issues and coverage gaps.
    """
    print("\n" + "=" * 80)
    print("  DATA AUDIT REPORT  —  Jio Narrative Shift Project")
    print("=" * 80)
    
    total = audit_dict.get("total_articles", 0)
    if total == 0:
        print("\n  [ERROR] Database is empty. Run Phase 1 & 2 first.")
        print("=" * 80 + "\n")
        return
    
    print(f"\n  Total Articles: {total}")
    
    date_range = audit_dict.get("date_range", (None, None))
    if date_range[0]:
        print(f"  Date Range: {date_range[0]} -> {date_range[1]}")
    
    # ── Coverage summary ──────────────────────────────────────────────────────
    weekly_df = audit_dict.get("articles_per_week", pd.DataFrame())
    if not weekly_df.empty:
        print(f"\n  Coverage:")
        print(f"    Weeks: {len(weekly_df)}")
        print(f"    Avg articles/week: {weekly_df['article_count'].mean():.1f}")
        print(f"    Min articles/week: {weekly_df['article_count'].min()}")
        print(f"    Max articles/week: {weekly_df['article_count'].max()}")
    
    # ── Coverage gaps ─────────────────────────────────────────────────────────
    gaps = audit_dict.get("coverage_gaps", [])
    if gaps:
        print(f"\n  [WARNING] Coverage Gaps ({len(gaps)} weeks with < 3 articles):")
        for w in gaps[:10]:
            print(f"      WARNING: Week {w}")
        if len(gaps) > 10:
            print(f"      ... and {len(gaps) - 10} more")
    else:
        print(f"\n  [OK] No coverage gaps (all weeks have >= 3 articles)")
    
    # ── Sources ───────────────────────────────────────────────────────────────
    source_df = audit_dict.get("source_breakdown", pd.DataFrame())
    if not source_df.empty:
        print(f"\n  Top Sources:")
        for idx, row in source_df.head(5).iterrows():
            print(f"    {row['source']}: {row['count']} ({row['pct']:.1f}%)")
    
    # ── Content quality ───────────────────────────────────────────────────────
    pct_missing = audit_dict.get("pct_content_missing", 0.0)
    pct_short = audit_dict.get("pct_short_content", 0.0)
    median_wc = audit_dict.get("median_word_count", 0.0)
    
    print(f"\n  Content Quality:")
    print(f"    Median word count: {median_wc:.0f}")
    print(f"    Missing content: {pct_missing:.1f}%")
    print(f"    Short content (< 50 words): {pct_short:.1f}%")
    
    # ── Critical warnings ─────────────────────────────────────────────────────
    if pct_missing > 10.0:
        print(f"\n  [CRITICAL] Over 10% of articles have missing content.")
    if len(gaps) > 5:
        print(f"\n  [CRITICAL] {len(gaps)} weeks with insufficient coverage.")
        print("      Consider: (a) backfill from GDELT, (b) interpolate JSD, (c) exclude period")
    
    print("\n" + "=" * 80 + "\n")


def detect_near_duplicates(conn: duckdb.DuckDBPyConnection,
                           similarity_threshold: float = 0.85,
                           max_titles: int = 1500) -> list:
    """
    Find near-duplicate titles using difflib.SequenceMatcher.
    Uses sampling to avoid O(n^2) explosion.
    """
    try:
        titles = conn.execute(
            "SELECT DISTINCT title FROM articles WHERE title IS NOT NULL"
        ).fetchall()

        titles = [t[0] for t in titles]

        # Limit size to avoid O(n^2) explosion
        if len(titles) > max_titles:
            logger.warning(
                "Too many titles (%d). Sampling first %d for duplicate detection.",
                len(titles), max_titles
            )
            titles = titles[:max_titles]

        duplicates = []

        for i, t1 in enumerate(titles):
            # Progress indicator
            if i % 200 == 0:
                logger.info("Duplicate scan progress: %d/%d", i, len(titles))

            for t2 in titles[i+1:]:
                ratio = SequenceMatcher(None, t1, t2).ratio()
                if ratio > similarity_threshold:
                    duplicates.append((t1, t2, ratio))

        if duplicates:
            logger.warning("Found %d near-duplicate title pairs", len(duplicates))
            for t1, t2, ratio in duplicates[:5]:
                logger.warning("  [%.3f] %s | %s", ratio, t1[:50], t2[:50])

        return duplicates

    except Exception as exc:
        logger.error("Failed to detect duplicates: %s", exc)
        return []


def plot_weekly_coverage(audit_dict: dict,
                        save_path: Optional[str] = None) -> None:
    """
    Create a bar chart of weekly article coverage.
    Red bars below threshold (3), blue bars above.
    
    Parameters
    ----------
    audit_dict : dict
        Output from run_audit()
    save_path : str, optional
        Where to save the chart. Default: plots/audit_coverage.png
    """
    if save_path is None:
        save_path = str(PLOTS_DIR / "audit_coverage.png")
    
    weekly_df = audit_dict.get("articles_per_week", pd.DataFrame())
    if weekly_df.empty:
        logger.warning("No weekly data to plot")
        return
    
    # Convert week_start to datetime if needed
    if not pd.api.types.is_datetime64_any_dtype(weekly_df["week_start"]):
        weekly_df = weekly_df.copy()
        weekly_df["week_start"] = pd.to_datetime(weekly_df["week_start"])
    
    # Sort chronologically
    weekly_df = weekly_df.sort_values("week_start").reset_index(drop=True)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 5))
    
    # Color bars: red if < 3, steelblue if >= 3
    colors = [
        "red" if count < 3 else "steelblue"
        for count in weekly_df["article_count"]
    ]
    
    ax.bar(weekly_df["week_start"], weekly_df["article_count"],
           color=colors, alpha=0.7, edgecolor="black", linewidth=0.5)
    
    # Minimum viable line
    ax.axhline(y=3, color="darkred", linestyle="--", linewidth=2, alpha=0.6,
               label="Minimum viable (3 articles)")
    
    # Formatting
    ax.set_xlabel("Week Start (Monday, ISO)", fontsize=11)
    ax.set_ylabel("Article Count", fontsize=11)
    ax.set_title("Weekly Article Coverage — Jio 2022–2024", fontsize=13, fontweight="bold")
    
    # Format x-axis
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    plt.xticks(rotation=45, ha="right")
    
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper right", fontsize=10)
    
    plt.tight_layout()
    
    # Create directory if needed
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    logger.info("Saved chart: %s", save_path)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info("=" * 80)
    logger.info("  Phase 3 — Data Audit")
    logger.info("=" * 80)
    
    # Open database
    if not DATA_DB_PATH.exists():
        logger.error("Database not found: %s", DATA_DB_PATH)
        logger.error("Run Phase 2 (storage.py) first.")
        exit(1)
    
    conn = duckdb.connect(str(DATA_DB_PATH), read_only=False)
    
    # Run audit
    audit_dict = run_audit(conn)
    
    # Print report
    print_audit_report(audit_dict)
    
    # Detect near-duplicates
    logger.info("\nChecking for near-duplicate titles...")
    duplicates = detect_near_duplicates(conn, similarity_threshold=0.85)
    if not duplicates:
        logger.info("✓ No significant near-duplicates found")
    
    # Generate plot
    logger.info("\nGenerating coverage chart...")
    plot_weekly_coverage(audit_dict)
    
    conn.close()
    logger.info("\nAudit complete.")
