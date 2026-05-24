"""
visualization.py — Phase 10: Visualization
==========================================
Generates publication-quality charts for the Narrative Shift Project.
Produces:
  1. plots/narrative_shift.png — weekly JSD shift score with z-score spikes
  2. plots/volatility.png — weekly RIL return volatility & rolling average
  3. plots/overlay.png — overlay of narrative shift vs. RIL stock volatility
  4. plots/topic_evolution.png — stacked area chart of top 5 Jio topic trends

Uses direct line annotations (no standard legend boxes) for a premium, clean look.
"""

import logging
import os
import pathlib
import sys
import warnings

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Suppress warnings
warnings.filterwarnings("ignore")

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
PLOTS_DIR = PROJECT_ROOT / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Style Helpers
# ---------------------------------------------------------------------------

def apply_style():
    """Apply premium matplotlib style settings for publications."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 13,
        "font.family": "DejaVu Sans",
        "figure.titlesize": 14,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "grid.alpha": 0.3,
    })


def save_plot(fig, path: str):
    """Save plot with 150 DPI and tight bounding box, then close figure."""
    path_obj = pathlib.Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path_obj), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved plot -> %s", path_obj)


# ---------------------------------------------------------------------------
# Plot 1: Narrative Shift Time Series
# ---------------------------------------------------------------------------

def plot_narrative_shift(shift_df: pd.DataFrame, spike_weeks: list, save_path: str = str(PLOTS_DIR / "narrative_shift.png")):
    """Plot weekly JSD shift score with z-score spike annotations."""
    apply_style()
    df = shift_df.copy()
    df["week_start"] = pd.to_datetime(df["week_start"])
    df = df.sort_values("week_start")

    fig, ax = plt.subplots(figsize=(14, 5))

    # Plot JSD score (thin gray line)
    ax.plot(df["week_start"], df["jsd_score"], color="gray", alpha=0.4, linewidth=1, label="JSD Raw")
    
    # Plot smoothed JSD score (thick blue line)
    ax.plot(df["week_start"], df["jsd_smoothed"], color="#1f77b4", alpha=0.9, linewidth=2.5, label="JSD Smoothed (3w)")

    # Plot vertical red dashed lines at spike weeks
    spike_datetimes = pd.to_datetime(spike_weeks)
    for idx, s_dt in enumerate(spike_datetimes):
        ax.axvline(s_dt, color="red", linestyle="--", alpha=0.5, linewidth=1.2)
        # Find local y-value to place text label slightly above the smoothed line
        row = df[df["week_start"] == s_dt]
        if not row.empty:
            y_val = row["jsd_smoothed"].values[0]
            if not np.isnan(y_val):
                ax.text(
                    s_dt, 
                    y_val + 0.02, 
                    s_dt.strftime("%Y-W%W"), 
                    color="darkred", 
                    fontsize=8, 
                    ha="center", 
                    va="bottom",
                    rotation=90,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.7)
                )

    # Format Date Axis
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.xticks(rotation=45, ha="right")

    # Direct line label (no legends)
    last_row = df.dropna(subset=["jsd_smoothed"]).iloc[-1]
    ax.annotate(
        "JSD Smoothed (3w)",
        xy=(last_row["week_start"], last_row["jsd_smoothed"]),
        xytext=(8, 0),
        textcoords="offset points",
        color="#1f77b4",
        fontweight="bold",
        va="center",
        fontsize=10
    )

    ax.set_title("Weekly Narrative Shift (JSD) — Jio News (2023–2025)", fontweight="bold", pad=15)
    ax.set_ylabel("Jensen-Shannon Divergence (JSD)")
    ax.set_xlabel("Timeline")
    ax.set_xlim(df["week_start"].min(), df["week_start"].max() + pd.Timedelta(days=45))  # extra room for label

    save_plot(fig, save_path)


# ---------------------------------------------------------------------------
# Plot 2: Financial Volatility
# ---------------------------------------------------------------------------

def plot_volatility(financial_df: pd.DataFrame, save_path: str = str(PLOTS_DIR / "volatility.png")):
    """Plot weekly stock return volatility and 4-week rolling average."""
    apply_style()
    df = financial_df.copy()
    df["week_start"] = pd.to_datetime(df["week_start"])
    df = df.sort_values("week_start")

    fig, ax = plt.subplots(figsize=(14, 5))

    # Plot volatility (thin gray line)
    ax.plot(df["week_start"], df["weekly_volatility"], color="gray", alpha=0.4, linewidth=1)
    
    # Plot rolling average (thick orange line)
    ax.plot(df["week_start"], df["rolling_vol_4w"], color="darkorange", alpha=0.9, linewidth=2.5)

    # Format Date Axis
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.xticks(rotation=45, ha="right")

    # Direct line labels
    last_row = df.dropna(subset=["rolling_vol_4w"]).iloc[-1]
    ax.annotate(
        "Rolling Volatility (4w)",
        xy=(last_row["week_start"], last_row["rolling_vol_4w"]),
        xytext=(8, 0),
        textcoords="offset points",
        color="darkorange",
        fontweight="bold",
        va="center",
        fontsize=10
    )

    ax.set_title("RELIANCE.NS Weekly Return Volatility (2023–2025)", fontweight="bold", pad=15)
    ax.set_ylabel("Volatility (Std Dev of Log Returns)")
    ax.set_xlabel("Timeline")
    ax.set_xlim(df["week_start"].min(), df["week_start"].max() + pd.Timedelta(days=50))

    save_plot(fig, save_path)


# ---------------------------------------------------------------------------
# Plot 3: Hero Overlay Chart
# ---------------------------------------------------------------------------

def plot_overlay(shift_df: pd.DataFrame, financial_df: pd.DataFrame, spike_weeks: list, save_path: str = str(PLOTS_DIR / "overlay.png")):
    """Generate dual y-axis overlay of narrative shift and stock volatility."""
    apply_style()
    
    # Load and clean data
    s_df = shift_df.copy()
    s_df["week_start"] = pd.to_datetime(s_df["week_start"])
    f_df = financial_df.copy()
    f_df["week_start"] = pd.to_datetime(f_df["week_start"])

    # Merge to align timelines for the overlay chart
    merged = pd.merge(s_df, f_df, on="week_start", how="inner").sort_values("week_start")

    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax2 = ax1.twinx()

    # Plot ax1 (left, blue): narrative shift
    color1 = "#1f77b4"
    ax1.plot(merged["week_start"], merged["jsd_smoothed"], color=color1, alpha=0.9, linewidth=2.5)
    ax1.set_ylabel("Narrative Shift (3w Smoothed JSD)", color=color1, fontweight="bold")
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(False) # Turn off grid for ax1 to prevent grid line collision

    # Plot ax2 (right, darkorange): stock volatility
    color2 = "darkorange"
    ax2.plot(merged["week_start"], merged["rolling_vol_4w"], color=color2, alpha=0.9, linewidth=2.5)
    ax2.set_ylabel("RIL Volatility (4w Rolling Std)", color=color2, fontweight="bold")
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.grid(True, alpha=0.2) # Keep background grid on secondary axis only

    # Draw vertical dashed red lines at spike weeks
    spike_dts = pd.to_datetime(spike_weeks)
    for s_dt in spike_dts:
        if merged["week_start"].min() <= s_dt <= merged["week_start"].max():
            ax1.axvline(s_dt, color="red", linestyle="--", alpha=0.35, linewidth=1.2)

    # Format Date Axis
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.sca(ax1)
    plt.xticks(rotation=45, ha="right")

    # Direct line labeling (no legend)
    last_row = merged.dropna(subset=["jsd_smoothed", "rolling_vol_4w"]).iloc[-1]
    ax1.annotate(
        "Narrative Shift",
        xy=(last_row["week_start"], last_row["jsd_smoothed"]),
        xytext=(8, -5),
        textcoords="offset points",
        color=color1,
        fontweight="bold",
        va="center",
        fontsize=9
    )
    ax2.annotate(
        "Stock Volatility",
        xy=(last_row["week_start"], last_row["rolling_vol_4w"]),
        xytext=(8, 5),
        textcoords="offset points",
        color=color2,
        fontweight="bold",
        va="center",
        fontsize=9
    )

    # Add text box for Granger results (the scientific conclusion)
    textstr = (
        "📊 Statistical Findings:\n"
        "• JSD -> Stock Volatility: Not Significant (p > 0.59)\n"
        "• Stock Volatility -> JSD (Reverse): Significant (p = 0.0334, Lag=2w)\n"
        "Conclusion: Stock price leads narrative focus (media is reactive)."
    )
    props = dict(boxstyle='round,pad=0.5', facecolor='#f7f7f7', edgecolor='lightgray', alpha=0.9)
    ax1.text(0.02, 0.95, textstr, transform=ax1.transAxes, fontsize=9.5,
            verticalalignment='top', bbox=props)

    ax1.set_title("Narrative Shift vs RIL Stock Volatility — Jio (2023–2025)", fontweight="bold", pad=20)
    ax1.set_xlabel("Timeline")
    ax1.set_xlim(merged["week_start"].min(), merged["week_start"].max() + pd.Timedelta(days=65))

    save_plot(fig, save_path)


# ---------------------------------------------------------------------------
# Plot 4: Stacked Topic Evolution over time
# ---------------------------------------------------------------------------

def plot_topic_evolution(topic_df: pd.DataFrame, save_path: str = str(PLOTS_DIR / "topic_evolution.png")):
    """Create stacked area chart showing how the top 5 topics evolved over time."""
    apply_style()
    df = topic_df.copy()
    
    # Exclude uncategorized outliers (-1)
    df = df[df["topic_id"] != -1]
    df["week_start"] = pd.to_datetime(df["week_start"])

    # 1. Identify the top 5 topic labels by total weight
    top5_topics = (
        df.groupby("topic_label")["weight"]
        .sum()
        .nlargest(5)
        .index
        .tolist()
    )
    
    # Filter to top 5 topics
    df_top5 = df[df["topic_label"].isin(top5_topics)]

    # 2. Pivot: rows = week_start, columns = topic_label, values = weight
    pivot_df = df_top5.pivot_table(
        index="week_start", 
        columns="topic_label", 
        values="weight", 
        aggfunc="sum"
    ).fillna(0.0)

    # Re-normalize to sum to 100% across the top 5 topics for visual balance
    row_sums = pivot_df.sum(axis=1)
    pivot_df = pivot_df.div(row_sums, axis=0).fillna(0.0)

    fig, ax = plt.subplots(figsize=(14, 7))

    # Plot stacked area chart
    weeks = pivot_df.index
    columns = pivot_df.columns.tolist()
    
    # Stack plots
    polys = ax.stackplot(
        weeks, 
        [pivot_df[c] for c in columns], 
        labels=columns, 
        colors=plt.cm.tab10.colors[:5],
        alpha=0.85,
        edgecolor="white",
        linewidth=0.3
    )

    # Format Date Axis
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.xticks(rotation=45, ha="right")

    # Direct Labeling of areas on the right edge
    # Compute the cumulative y-midpoints at the last index
    last_vals = pivot_df.iloc[-1].values
    cum_vals = np.cumsum(last_vals)
    prev_val = 0.0

    for idx, col_name in enumerate(columns):
        y_mid = (cum_vals[idx] + prev_val) / 2.0
        prev_val = cum_vals[idx]

        # Use the name of the topic but format it slightly nicer (clean underscores)
        clean_label = col_name.replace("_", " ").title()
        
        ax.annotate(
            clean_label,
            xy=(weeks[-1], y_mid),
            xytext=(10, 0),
            textcoords="offset points",
            color=plt.cm.tab10.colors[idx],
            fontweight="bold",
            va="center",
            fontsize=9.5
        )

    ax.set_title("Topic Evolution (Top 5 Evolving Narratives) — Jio (2023–2025)", fontweight="bold", pad=20)
    ax.set_ylabel("Relative Topic Weight (Share of Voice)")
    ax.set_xlabel("Timeline")
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    ax.set_xlim(weeks.min(), weeks.max() + pd.Timedelta(days=80)) # extra room for annotations

    save_plot(fig, save_path)


# ---------------------------------------------------------------------------
# Pipeline Orchestrator
# ---------------------------------------------------------------------------

def run_all_plots():
    """Load parquet outputs and generate all 4 core visual charts."""
    shift_path = DATA_DIR / "narrative_shift.parquet"
    financial_path = DATA_DIR / "financial_metrics.parquet"
    topic_path = DATA_DIR / "topic_distributions.parquet"

    # Verify files
    if not (shift_path.exists() and financial_path.exists() and topic_path.exists()):
        logger.error(
            "Missing input Parquet files. Ensure topic modeling, shift detection, "
            "and finance modules have been run first."
        )
        sys.exit(1)

    # 1. Load Data
    logger.info("Loading dataset parquets …")
    shift_df = pd.read_parquet(shift_path)
    financial_df = pd.read_parquet(financial_path)
    topic_df = pd.read_parquet(topic_path)

    # 2. Get Spike Weeks
    from shift_detection import identify_spike_weeks
    spike_weeks = identify_spike_weeks(shift_df, z_threshold=1.5)

    # 3. Generate plots
    logger.info("Generating plot charts …")
    plot_narrative_shift(shift_df, spike_weeks)
    plot_volatility(financial_df)
    plot_overlay(shift_df, financial_df, spike_weeks)
    plot_topic_evolution(topic_df)

    logger.info("✓ All Phase 10 plots generated successfully!")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_all_plots()
    sys.exit(0)
