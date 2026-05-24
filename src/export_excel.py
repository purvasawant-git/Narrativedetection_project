"""
export_excel.py — Export Project Data to Excel for Power BI Ingestion
=====================================================================
Consolidates the outputs of all pipeline phases into a single, multi-sheet Excel
file: data/project_outputs.xlsx.

Sheets exported:
  1. Articles_Summary       — Summary metrics from DuckDB (top sources, keywords, quality)
  2. Topic_Distributions    — Weekly BERTopic topic distributions and weights
  3. Narrative_Shifts       — Weekly JSD narrative shift scores & smoothed scores
  4. Financial_Metrics      — Weekly close returns, stock volatility & NIFTY returns
  5. Aligned_Dataset        — Cleaned, stationary aligned dataset
  6. Granger_Results        — Output of Granger causality tests (lags, directions, p-values)
  7. News_Planner_Data      — Volatility Z-scores matched to shifted (next week) JSD and PR playbooks
  8. Calm_vs_Volatile_Topics — Topic distribution comparisons between Calm and Volatile market states
"""

import logging
import pathlib
import sys
import warnings

import duckdb
import numpy as np
import pandas as pd

# Suppress openpyxl styling warnings if any
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "articles.duckdb"
OUTPUT_EXCEL = DATA_DIR / "project_outputs.xlsx"


# ---------------------------------------------------------------------------
# Helpers for DuckDB summaries
# ---------------------------------------------------------------------------

def get_db_summaries(db_path: pathlib.Path) -> dict:
    """Query DuckDB database to return summary DataFrames for sheet 1."""
    if not db_path.exists():
        logger.warning("articles.duckdb not found. Skipping DB summaries.")
        return {}

    conn = duckdb.connect(str(db_path), read_only=True)
    summaries = {}

    try:
        # 1. Overall stats
        stats_df = conn.execute("""
            SELECT
                (SELECT COUNT(*) FROM articles) AS total_articles,
                MIN(published_at) AS first_article_date,
                MAX(published_at) AS last_article_date,
                PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY word_count) AS median_words,
                (COUNT(*) FILTER (WHERE content IS NULL OR content = '') * 100.0 / COUNT(*)) AS pct_missing_content,
                (COUNT(*) FILTER (WHERE word_count < 50) * 100.0 / COUNT(*)) AS pct_short_articles
            FROM articles
        """).df()
        
        # Format dates to string
        stats_df["first_article_date"] = stats_df["first_article_date"].astype(str)
        stats_df["last_article_date"] = stats_df["last_article_date"].astype(str)
        
        # Transpose for easier reading in Power BI / Excel
        overall_stats = stats_df.T.reset_index()
        overall_stats.columns = ["Metric", "Value"]
        summaries["overall_stats"] = overall_stats

        # 2. Top Sources
        top_sources = conn.execute("""
            SELECT 
                source, 
                COUNT(*) AS article_count,
                ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM articles), 2) AS percentage
            FROM articles
            WHERE source IS NOT NULL AND source != ''
            GROUP BY source
            ORDER BY article_count DESC
            LIMIT 15
        """).df()
        summaries["top_sources"] = top_sources

        # 3. Top Keyword Groups
        top_keywords = conn.execute("""
            SELECT 
                keyword_group, 
                COUNT(*) AS article_count,
                ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM articles), 2) AS percentage
            FROM articles
            WHERE keyword_group IS NOT NULL AND keyword_group != ''
            GROUP BY keyword_group
            ORDER BY article_count DESC
            LIMIT 15
        """).df()
        summaries["top_keywords"] = top_keywords

    except Exception as exc:
        logger.error("Failed to query DuckDB summaries: %s", exc)
    finally:
        conn.close()

    return summaries


# ---------------------------------------------------------------------------
# Main exporter
# ---------------------------------------------------------------------------

def export_to_excel():
    """Load all parquets and compile them with DuckDB summaries into one Excel file."""
    logger.info("Initializing Excel export...")

    # Data files dict
    parquets = {
        "Topic_Distributions": DATA_DIR / "topic_distributions.parquet",
        "Narrative_Shifts": DATA_DIR / "narrative_shift.parquet",
        "Financial_Metrics": DATA_DIR / "financial_metrics.parquet",
        "Aligned_Dataset": DATA_DIR / "aligned_final.parquet",
        "Granger_Results": DATA_DIR / "granger_results.parquet"
    }

    # Verify DATA_DIR
    if not DATA_DIR.exists():
        logger.error("Data directory does not exist: %s", DATA_DIR)
        sys.exit(1)

    # Initialize ExcelWriter
    logger.info("Writing Excel file -> %s", OUTPUT_EXCEL)
    with pd.ExcelWriter(OUTPUT_EXCEL, engine="openpyxl") as writer:
        
        # --- Sheet 1: Articles Summary & Stats (Split into 3 for clean Power BI import) ---
        logger.info("Generating Articles Summary sheets...")
        db_data = get_db_summaries(DB_PATH)
        if db_data:
            db_data["overall_stats"].to_excel(writer, sheet_name="Summary_Overall_Stats", index=False)
            db_data["top_sources"].to_excel(writer, sheet_name="Summary_Top_Sources", index=False)
            db_data["top_keywords"].to_excel(writer, sheet_name="Summary_Top_Keywords", index=False)
        else:
            pd.DataFrame({"Status": ["DuckDB articles summary not available"]}).to_excel(
                writer, sheet_name="Summary_Overall_Stats", index=False
            )

        # --- Sheets 2-6: Parquet Outputs ---
        for sheet_name, p_path in parquets.items():
            if p_path.exists():
                logger.info("Loading and writing sheet: %s...", sheet_name)
                df = pd.read_parquet(p_path)
                
                # Format datetime columns to string (so Excel parses them nicely and cleanly)
                for col in df.columns:
                    if pd.api.types.is_datetime64_any_dtype(df[col]):
                        df[col] = df[col].astype(str)
                
                df.to_excel(writer, sheet_name=sheet_name, index=False)
            else:
                logger.warning("Skipping sheet %s: file %s does not exist.", sheet_name, p_path.name)
                pd.DataFrame({"Status": [f"Data file '{p_path.name}' not found"]}).to_excel(
                    writer, sheet_name=sheet_name, index=False
                )

        # --- Sheet 7: News Planner Data (Predictive Dashboard source) ---
        logger.info("Generating News_Planner_Data sheet...")
        aligned_path = DATA_DIR / "aligned_final.parquet"
        if aligned_path.exists():
            aligned_df = pd.read_parquet(aligned_path)
            aligned_df["week_start"] = pd.to_datetime(aligned_df["week_start"])
            
            vol_mean = aligned_df["volatility_final"].mean()
            vol_std = aligned_df["volatility_final"].std()
            aligned_df["volatility_z_score"] = (aligned_df["volatility_final"] - vol_mean) / vol_std
            
            # Segment market states based on Z-score
            aligned_df["market_state"] = np.where(
                aligned_df["volatility_z_score"] > 1.0, 
                "Volatile (High Risk)", 
                "Calm (Standard)"
            )
            
            # Shift JSD forward: JSD of week t+1 and week t+2 matched to current week t volatility
            aligned_df["next_week_predicted_jsd"] = aligned_df["jsd_final"].shift(-1)
            aligned_df["week_after_next_predicted_jsd"] = aligned_df["jsd_final"].shift(-2)
            
            # Actionable PR playbook strategy recommendations
            aligned_df["recommended_pr_playbook"] = np.where(
                aligned_df["volatility_z_score"] > 1.0,
                "Crisis & Financial Disclosure Strategy (Proactively address market rumors, focus on liquidity/debt metrics)",
                "Standard Marketing & Product Strategy (Promote customer rollouts, network achievements, CSR, and features)"
            )
            
            # Format datetime
            aligned_df["week_start"] = aligned_df["week_start"].astype(str)
            aligned_df.to_excel(writer, sheet_name="News_Planner_Data", index=False)
        else:
            logger.warning("aligned_final.parquet not found. Skipping News_Planner_Data.")

        # --- Sheet 8: Calm vs Volatile Topics ---
        logger.info("Generating Calm_vs_Volatile_Topics sheet...")
        topic_path = DATA_DIR / "topic_distributions.parquet"
        if aligned_path.exists() and topic_path.exists():
            topic_df = pd.read_parquet(topic_path)
            topic_df["week_start"] = pd.to_datetime(topic_df["week_start"])
            
            state_df = pd.read_parquet(aligned_path)
            state_df["week_start"] = pd.to_datetime(state_df["week_start"])
            
            v_mean = state_df["volatility_final"].mean()
            v_std = state_df["volatility_final"].std()
            state_df["volatility_z_score"] = (state_df["volatility_final"] - v_mean) / v_std
            state_df["market_state"] = np.where(state_df["volatility_z_score"] > 1.0, "Volatile", "Calm")
            
            topic_merged = pd.merge(topic_df, state_df[["week_start", "market_state"]], on="week_start", how="inner")
            
            # Exclude outliers (-1)
            topic_merged = topic_merged[topic_merged["topic_id"] != -1]
            
            # Average weight grouped by topic and market state
            grouped = topic_merged.groupby(["topic_label", "market_state"])["weight"].mean().unstack(fill_value=0.0)
            
            # Check for column existence
            for col in ["Calm", "Volatile"]:
                if col not in grouped.columns:
                    grouped[col] = 0.0
            
            grouped = grouped.rename(columns={"Calm": "avg_weight_calm_weeks", "Volatile": "avg_weight_volatile_weeks"})
            grouped["weight_change"] = grouped["avg_weight_volatile_weeks"] - grouped["avg_weight_calm_weeks"]
            
            # Ratio of increase
            grouped["weight_increase_ratio"] = np.where(
                grouped["avg_weight_calm_weeks"] > 0,
                grouped["avg_weight_volatile_weeks"] / grouped["avg_weight_calm_weeks"],
                grouped["avg_weight_volatile_weeks"] / 0.0001
            )
            
            grouped = grouped.reset_index().sort_values(by="weight_change", ascending=False)
            grouped.to_excel(writer, sheet_name="Calm_vs_Volatile_Topics", index=False)
        else:
            logger.warning("Skipping Calm_vs_Volatile_Topics due to missing files.")

    logger.info("✓ Excel compilation complete! Saved to %s", OUTPUT_EXCEL)


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    export_to_excel()
    sys.exit(0)
