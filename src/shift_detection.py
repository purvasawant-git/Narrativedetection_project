"""
shift_detection.py — Phase 6: Narrative Shift Detection
========================================================
Project : NarrativeShiftProject (Reliance Jio)
Input   : data/topic_distributions.parquet  (from Phase 5 topic_model.py)
Output  : data/narrative_shift.parquet

WHY JSD OVER NMI:
    NMI measures mutual information between hard cluster assignments — it requires
    discretizing distributions into class labels, which loses information.
    JSD (Jensen-Shannon Divergence) compares soft probability distributions
    directly, capturing the full shape of each week's topic mix without any
    discretization loss. JSD is symmetric, bounded in [0, 1], and well-defined
    even when distributions have non-overlapping support. It is the right metric
    for comparing consecutive weekly topic distributions.

IMPORTANT — Topic alignment note:
    Because Phase 5 runs an independent BERTopic model per week, topic_id=0 in
    week 1 is NOT semantically equivalent to topic_id=0 in week 2. The pivot in
    build_distribution_matrix treats each (week, topic_id) column as an independent
    dimension. This is intentional: we are measuring whether this week's TOPIC MIX
    (its weight vector across its own topic IDs) looks similar in shape to the
    previous week's mix — not whether the same named topics recur.
"""

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("shift_detection")

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
INPUT_PATH = DATA_DIR / "topic_distributions.parquet"
OUTPUT_PATH = DATA_DIR / "narrative_shift.parquet"


# ---------------------------------------------------------------------------
# 1. build_distribution_matrix
# ---------------------------------------------------------------------------

def build_distribution_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot topic_distributions into a W × T probability matrix.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: week_start (str), topic_id (int), weight (float).

    Returns
    -------
    pd.DataFrame
        Rows = week_start sorted chronologically (str index).
        Columns = topic_id (int).
        Each row sums to 1.0 (re-normalized after pivot to correct float drift).
        Missing (week, topic) pairs are filled with 0.
    """
    logger.info("Building distribution matrix from %d (week, topic) pairs …", len(df))

    # Aggregate in case there are duplicate (week_start, topic_id) rows
    agg = (
        df.groupby(["week_start", "topic_id"], as_index=False)["weight"]
        .sum()
    )

    # Pivot: rows = week, columns = topic_id, values = summed weight
    matrix = agg.pivot(index="week_start", columns="topic_id", values="weight")
    matrix.columns.name = None          # clean up MultiIndex label
    matrix.index.name = "week_start"

    # Fill missing topic slots with 0
    matrix = matrix.fillna(0.0)

    # Sort chronologically
    matrix = matrix.sort_index()

    # Row-wise re-normalization to sum = 1.0 (fixes any float drift post-pivot)
    row_sums = matrix.sum(axis=1)
    zero_rows = row_sums[row_sums == 0]
    if not zero_rows.empty:
        warnings.warn(
            f"Weeks with all-zero weight after pivot (will remain zero): "
            f"{zero_rows.index.tolist()}",
            RuntimeWarning,
        )
    # Avoid division by zero; rows that sum to 0 stay 0
    matrix = matrix.div(row_sums.replace(0, np.nan), axis=0).fillna(0.0)

    logger.info(
        "Distribution matrix shape: %d weeks × %d topics", *matrix.shape
    )
    logger.debug("Week range: %s → %s", matrix.index[0], matrix.index[-1])
    return matrix


# ---------------------------------------------------------------------------
# 2. jensen_shannon_divergence
# ---------------------------------------------------------------------------

def jensen_shannon_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """
    Compute the Jensen-Shannon Divergence between two probability vectors.

    Uses scipy.spatial.distance.jensenshannon which returns the *square root*
    of JSD (the JS distance). We square it to obtain the proper divergence:
        JSD(p || q) = jensenshannon(p, q) ** 2

    Parameters
    ----------
    p, q : np.ndarray
        Non-negative weight vectors. Need not sum to 1 — scipy normalizes internally.

    Returns
    -------
    float
        JSD in [0, 1], or np.nan if either array is all-zeros or contains NaN.
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)

    # Guard: NaN in either vector
    if np.any(np.isnan(p)) or np.any(np.isnan(q)):
        return np.nan

    # Guard: all-zero vector (no distribution to compare)
    if np.all(p == 0) or np.all(q == 0):
        return np.nan

    # scipy returns JS *distance* = sqrt(JSD); square to get divergence
    js_distance = jensenshannon(p, q)
    jsd = float(js_distance ** 2)
    return jsd


# ---------------------------------------------------------------------------
# 3. compute_shift_scores
# ---------------------------------------------------------------------------

def compute_shift_scores(dist_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Compute week-over-week JSD for every consecutive pair of weeks.

    Parameters
    ----------
    dist_matrix : pd.DataFrame
        Output of build_distribution_matrix (W × T, rows sorted chronologically).

    Returns
    -------
    pd.DataFrame
        Columns:
            week_start   (str)   — the LATER week of the consecutive pair
            jsd_score    (float) — JSD between week i and week i+1
            jsd_smoothed (float) — 3-week rolling mean of jsd_score (min_periods=2)
    """
    weeks = dist_matrix.index.tolist()
    n = len(weeks)

    if n < 2:
        raise ValueError(
            f"Need at least 2 weeks to compute shift scores; got {n}."
        )

    logger.info("Computing JSD for %d consecutive week pairs …", n - 1)

    records = []
    for i in range(n - 1):
        p = dist_matrix.iloc[i].values
        q = dist_matrix.iloc[i + 1].values
        jsd = jensen_shannon_divergence(p, q)
        records.append({"week_start": weeks[i + 1], "jsd_score": jsd})

    shift_df = pd.DataFrame(records)

    # 3-week rolling mean of jsd_score (look-back window over the later week)
    shift_df["jsd_smoothed"] = (
        shift_df["jsd_score"]
        .rolling(window=3, min_periods=2)
        .mean()
    )

    nan_count = shift_df["jsd_score"].isna().sum()
    if nan_count:
        logger.warning("%d week pairs returned NaN JSD (zero-weight rows).", nan_count)

    logger.info(
        "Shift scores — mean=%.4f  std=%.4f  max=%.4f",
        shift_df["jsd_score"].mean(),
        shift_df["jsd_score"].std(),
        shift_df["jsd_score"].max(),
    )
    return shift_df


# ---------------------------------------------------------------------------
# 4. identify_spike_weeks
# ---------------------------------------------------------------------------

def identify_spike_weeks(
    shift_df: pd.DataFrame,
    z_threshold: float = 1.5,
) -> list:
    """
    Identify weeks whose JSD score is a statistical spike.

    Parameters
    ----------
    shift_df : pd.DataFrame
        Output of compute_shift_scores.
    z_threshold : float
        Z-score cut-off above which a week is flagged as a spike (default 1.5).

    Returns
    -------
    list[str]
        week_start strings where z_score > z_threshold, sorted chronologically.
    """
    scores = shift_df["jsd_score"].dropna()

    if scores.empty:
        logger.warning("No valid JSD scores available; returning empty spike list.")
        return []

    mean_jsd = scores.mean()
    std_jsd = scores.std(ddof=1)

    if std_jsd == 0:
        logger.warning("JSD std=0; no spikes can be identified.")
        return []

    shift_df = shift_df.copy()
    shift_df["z_score"] = (shift_df["jsd_score"] - mean_jsd) / std_jsd

    spike_mask = shift_df["z_score"] > z_threshold
    spike_weeks = shift_df.loc[spike_mask, "week_start"].tolist()

    logger.info(
        "Spike detection (z > %.2f): %d spike week(s) identified out of %d.",
        z_threshold,
        len(spike_weeks),
        len(shift_df),
    )

    # Log top 5 highest-shift weeks
    top5 = (
        shift_df.dropna(subset=["jsd_score"])
        .nlargest(5, "jsd_score")[["week_start", "jsd_score", "z_score"]]
    )
    logger.info("── Top 5 highest-shift weeks ──────────────────────────────")
    for _, row in top5.iterrows():
        try:
            iso_week = pd.Timestamp(row["week_start"]).strftime("%Y-W%W")
        except Exception:
            iso_week = row["week_start"]
        flag = "SPIKE" if row["z_score"] > z_threshold else "     "
        logger.info(
            "%s: Week %s (%s) | JSD=%.4f (z=%.2f) — transition logged",
            flag,
            iso_week,
            row["week_start"],
            row["jsd_score"],
            row["z_score"],
        )
    logger.info("────────────────────────────────────────────────────────────")

    return sorted(spike_weeks)


# ---------------------------------------------------------------------------
# 5. run_full_pipeline
# ---------------------------------------------------------------------------

def run_full_pipeline() -> pd.DataFrame:
    """
    End-to-end pipeline: load → build matrix → compute shifts → detect spikes → save.

    Returns
    -------
    pd.DataFrame
        shift_df with columns: week_start, jsd_score, jsd_smoothed.
        (z_score is computed internally during spike detection but not persisted.)
    """
    logger.info("═" * 60)
    logger.info("Phase 6 — Narrative Shift Detection  START")
    logger.info("═" * 60)

    # ── Load ──────────────────────────────────────────────────────────────
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_PATH}\n"
            "Ensure Phase 5 (topic_model.py) has been run first."
        )
    logger.info("Loading topic distributions from: %s", INPUT_PATH)
    df = pd.read_parquet(INPUT_PATH)
    logger.info(
        "Loaded %d rows | %d unique weeks | %d unique topics",
        len(df),
        df["week_start"].nunique(),
        df["topic_id"].nunique(),
    )

    # ── Build matrix ──────────────────────────────────────────────────────
    dist_matrix = build_distribution_matrix(df)

    # ── Compute shift scores ───────────────────────────────────────────────
    shift_df = compute_shift_scores(dist_matrix)

    # ── Identify spikes ────────────────────────────────────────────────────
    spike_weeks = identify_spike_weeks(shift_df, z_threshold=1.5)
    logger.info("Spike weeks (%d total): %s", len(spike_weeks), spike_weeks)

    # ── Save output (merge with existing if present) ─────────────────────
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if OUTPUT_PATH.exists():
        existing = pd.read_parquet(OUTPUT_PATH)
        shift_df = pd.concat([existing, shift_df], ignore_index=True)
        shift_df.drop_duplicates(subset=["week_start"], keep="last", inplace=True)
        shift_df.sort_values("week_start", inplace=True)
        shift_df.reset_index(drop=True, inplace=True)
        logger.info("Merged with existing narrative_shift.parquet")
    shift_df.to_parquet(OUTPUT_PATH, index=False)
    logger.info("Saved narrative_shift.parquet → %s", OUTPUT_PATH)
    logger.info("Output schema: %s", dict(shift_df.dtypes))
    logger.info("═" * 60)
    logger.info("Phase 6 — Narrative Shift Detection  COMPLETE")
    logger.info("═" * 60)

    return shift_df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    result = run_full_pipeline()
    print("\n── Shift Detection Results (first 10 rows) ──")
    print(result.head(10).to_string(index=False))
    print(f"\nTotal weeks scored : {len(result)}")
    print(f"Mean JSD           : {result['jsd_score'].mean():.4f}")
    print(f"Max  JSD           : {result['jsd_score'].max():.4f}")
    print(f"Saved to           : {OUTPUT_PATH}")
