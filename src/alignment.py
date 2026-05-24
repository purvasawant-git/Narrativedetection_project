"""
alignment.py — Phase 8: Alignment & Stationarity
================================================
Merges the weekly narrative shift time series (Phase 6) and weekly financial
metrics (Phase 7) on week_start. Tests the merged series for stationarity using
ADF and KPSS tests. If either series is non-stationary, applies first-differencing
to ensure valid Granger causality testing. Saves the result to data/aligned_final.parquet.

WHY THIS MATTERS:
  Granger causality is a regression-based test. If jsd_score has an upward trend 
  (narratives generally getting more volatile over 2 years), it will "predict" 
  anything else that trends upward — including stock volatility — not because of 
  causal structure but because of shared trends. First-differencing removes the trend
  and leaves only week-to-week changes, which is what we actually want to test.
"""

import logging
import pathlib
import sys
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss

# Suppress statsmodels interpolation warnings for KPSS
warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------

def merge_time_series(shift_df: pd.DataFrame, financial_df: pd.DataFrame) -> pd.DataFrame:
    """Convert shift_df.week_start to datetime if string, join onto financial_df,

    and clean.

    Parameters
    ----------
    shift_df     : pd.DataFrame (JSD shift scores from Phase 6)
    financial_df : pd.DataFrame (Stock metrics from Phase 7)

    Returns
    -------
    pd.DataFrame
        Merged, cleaned, and sorted dataframe.
    """
    # 1. Convert week_start to datetime if string
    shift_df = shift_df.copy()
    financial_df = financial_df.copy()

    shift_df["week_start"] = pd.to_datetime(shift_df["week_start"])
    financial_df["week_start"] = pd.to_datetime(financial_df["week_start"])

    # 2. Left join shift_df onto financial_df on week_start
    merged = pd.merge(financial_df, shift_df, on="week_start", how="left")

    # 3. Sort by week_start ascending
    merged = merged.sort_values("week_start").reset_index(drop=True)

    # 4. Forward-fill financial columns by up to 1 week (market holiday gaps)
    financial_cols = [
        "weekly_return",
        "weekly_volatility",
        "rolling_vol_4w",
        "nifty_return",
    ]
    cols_to_ffill = [c for c in financial_cols if c in merged.columns]
    merged[cols_to_ffill] = merged[cols_to_ffill].ffill(limit=1)

    # 5. Drop rows where EITHER jsd_score OR weekly_volatility is NaN
    merged = merged.dropna(subset=["jsd_score", "weekly_volatility"]).reset_index(drop=True)

    # 6. Log merge results
    min_date = merged["week_start"].min().strftime("%Y-%m-%d")
    max_date = merged["week_start"].max().strftime("%Y-%m-%d")
    logger.info(
        "Merged dataset: %d weeks, date range %s to %s",
        len(merged),
        min_date,
        max_date,
    )

    return merged


def check_stationarity(series: pd.Series, name: str) -> dict:
    """Run ADF and KPSS tests on a series to determine stationarity.

    - ADF H0: Unit root (non-stationary). p < 0.05 -> Reject H0 -> Stationary.
    - KPSS H0: Trend/Level stationary. p > 0.05 -> Fail to reject -> Stationary.

    Parameters
    ----------
    series : pd.Series
    name   : str

    Returns
    -------
    dict
        Contains name, adf_pvalue, kpss_pvalue, is_stationary.
    """
    clean_series = series.dropna()

    # ADF Test
    adf_result = adfuller(clean_series, autolag="AIC")
    adf_pvalue = float(adf_result[1])

    # KPSS Test
    kpss_result = kpss(clean_series, regression="c", nlags="auto")
    kpss_pvalue = float(kpss_result[1])

    # Stationary only if both tests indicate stationarity
    is_stationary = (adf_pvalue < 0.05) and (kpss_pvalue > 0.05)

    status = "STATIONARY" if is_stationary else "NOT STATIONARY"
    print(f"{name}: ADF p={adf_pvalue:.4f}, KPSS p={kpss_pvalue:.4f} -> {status}")

    return {
        "name": name,
        "adf_pvalue": adf_pvalue,
        "kpss_pvalue": kpss_pvalue,
        "is_stationary": is_stationary,
    }


def make_stationary(series: pd.Series, name: str) -> pd.Series:
    """If not stationary, apply first difference (series.diff().dropna()).

    Otherwise, return series unchanged.
    """
    diag = check_stationarity(series, name)
    if not diag["is_stationary"]:
        logger.info("Applied first-differencing to %s", name)
        return series.diff().dropna()
    else:
        return series


def prepare_for_granger(aligned_df: pd.DataFrame) -> pd.DataFrame:
    """Diagnose stationarity, differencing non-stationary series, and format output.

    Parameters
    ----------
    aligned_df : pd.DataFrame
        Output of merge_time_series.

    Returns
    -------
    pd.DataFrame
        Aligned dataframe ready for Granger.
    """
    # 1. Run check_stationarity and make_stationary on JSD and Volatility
    jsd_final = make_stationary(aligned_df["jsd_score"], "jsd_score")
    volatility_final = make_stationary(aligned_df["weekly_volatility"], "weekly_volatility")

    # 2. Check if transformations were applied
    jsd_diffed = not jsd_final.equals(aligned_df["jsd_score"])
    vol_diffed = not volatility_final.equals(aligned_df["weekly_volatility"])

    # 3. Construct final DataFrame
    final_df = pd.DataFrame(
        {
            "week_start": aligned_df["week_start"],
            "jsd_final": jsd_final,
            "volatility_final": volatility_final,
            "nifty_return": aligned_df["nifty_return"],
        }
    )

    # 4. Drop rows where either is NaN (due to differencing dropna)
    final_df = final_df.dropna(subset=["jsd_final", "volatility_final"]).reset_index(drop=True)

    # 5. Add metadata attributes
    final_df.attrs["transformations_applied"] = {
        "jsd_differenced": jsd_diffed,
        "volatility_differenced": vol_diffed,
    }

    return final_df


def run_full_pipeline(
    shift_path: pathlib.Path | None = None,
    financial_path: pathlib.Path | None = None,
    output_path: pathlib.Path | None = None,
) -> pd.DataFrame:
    """Load parquet outputs, merge them, prepare for Granger, and save results."""
    if shift_path is None:
        shift_path = DATA_DIR / "narrative_shift.parquet"
    if financial_path is None:
        financial_path = DATA_DIR / "financial_metrics.parquet"
    if output_path is None:
        output_path = DATA_DIR / "aligned_final.parquet"

    # Verify input file existence
    if not shift_path.exists():
        raise FileNotFoundError(f"Missing shift metrics: {shift_path}")
    if not financial_path.exists():
        raise FileNotFoundError(f"Missing financial metrics: {financial_path}")

    # Load Parquets
    shift_df = pd.read_parquet(shift_path)
    financial_df = pd.read_parquet(financial_path)

    # Merge
    merged_df = merge_time_series(shift_df, financial_df)

    # Run check & differencing
    final_df = prepare_for_granger(merged_df)

    # Save output
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    final_df.to_parquet(output_path, index=False)
    logger.info("Saved final aligned dataset → %s", output_path)

    return final_df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_full_pipeline()
    sys.exit(0)
