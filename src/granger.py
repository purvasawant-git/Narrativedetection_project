"""
granger.py — Phase 9: Granger Causality Testing
==============================================
Runs bivariate Granger causality tests and multivariate Vector Autoregression (VAR)
Granger tests (controlled for NIFTY index returns) to determine if weekly
narrative shifts (JSD scores) predict RIL stock volatility.

Saves results to data/granger_results.parquet.
"""

import logging
import pathlib
import sys
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.vector_ar.var_model import VAR

# Suppress statsmodels warnings if any
warnings.filterwarnings("ignore", category=FutureWarning, module="statsmodels")

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
INPUT_PATH = DATA_DIR / "aligned_final.parquet"
OUTPUT_PATH = DATA_DIR / "granger_results.parquet"


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------

def run_granger_test(df: pd.DataFrame, max_lag: int = 4) -> pd.DataFrame:
    """Run standard bivariate Granger causality tests in both directions.

    Direction 1: jsd_final -> volatility_final
    Direction 2: volatility_final -> jsd_final (reverse)

    Parameters
    ----------
    df      : pd.DataFrame
    max_lag : int, default 4

    Returns
    -------
    pd.DataFrame
        Columns: lag, direction, f_stat, p_value, significant, controlled
    """
    results = []

    # 1. Direction: JSD -> Volatility (Test: does JSD predict Volatility?)
    # statsmodels expects 2D array where col 1 is dependent (Y) and col 2 is independent (X)
    logger.info("Running bivariate Granger tests: JSD -> Volatility")
    try:
        res_dir1 = grangercausalitytests(
            df[["volatility_final", "jsd_final"]],
            maxlag=max_lag,
            verbose=False,
        )
        for lag in range(1, max_lag + 1):
            f_stat, p_val, _, _ = res_dir1[lag][0]["ssr_ftest"]
            results.append({
                "lag": lag,
                "direction": "jsd -> volatility",
                "f_stat": float(f_stat),
                "p_value": float(p_val),
                "significant": bool(p_val < 0.05),
                "controlled": False
            })
    except Exception as exc:
        logger.error("Failed Granger test (JSD -> Volatility): %s", exc)

    # 2. Reverse Direction: Volatility -> JSD (Test: does Volatility predict JSD?)
    logger.info("Running bivariate Granger tests: Volatility -> JSD (Reverse)")
    try:
        res_dir2 = grangercausalitytests(
            df[["jsd_final", "volatility_final"]],
            maxlag=max_lag,
            verbose=False,
        )
        for lag in range(1, max_lag + 1):
            f_stat, p_val, _, _ = res_dir2[lag][0]["ssr_ftest"]
            results.append({
                "lag": lag,
                "direction": "volatility -> jsd",
                "f_stat": float(f_stat),
                "p_value": float(p_val),
                "significant": bool(p_val < 0.05),
                "controlled": False
            })
    except Exception as exc:
        logger.error("Failed Granger test (Volatility -> JSD): %s", exc)

    return pd.DataFrame(results)


def run_controlled_granger(df: pd.DataFrame, max_lag: int = 4) -> pd.DataFrame:
    """Run multivariate Granger causality test using Vector Autoregression (VAR).

    Controls for NIFTY index returns to verify if JSD predicts RIL volatility
    beyond general market movements.

    Parameters
    ----------
    df      : pd.DataFrame
    max_lag : int, default 4

    Returns
    -------
    pd.DataFrame
        Columns: lag, direction, f_stat, p_value, significant, controlled
    """
    logger.info("Running controlled Granger tests using VAR model")
    results = []

    # Columns of interest for the VAR model
    var_cols = ["jsd_final", "volatility_final", "nifty_return"]
    var_df = df[var_cols].copy()

    for lag in range(1, max_lag + 1):
        try:
            model = VAR(var_df)
            fitted_model = model.fit(lag)
            
            # Test if jsd_final Granger-causes volatility_final
            test_res = fitted_model.test_causality(
                "volatility_final",
                ["jsd_final"],
                kind="f"
            )
            
            f_stat = float(test_res.test_statistic)
            p_val = float(test_res.pvalue)
            
            results.append({
                "lag": lag,
                "direction": "jsd -> volatility",
                "f_stat": f_stat,
                "p_value": p_val,
                "significant": bool(p_val < 0.05),
                "controlled": True
            })
        except Exception as exc:
            logger.error("Failed controlled Granger test at lag %d: %s", lag, exc)

    return pd.DataFrame(results)


def interpret_results(granger_df: pd.DataFrame) -> str:
    """Generate a plain-English statistical interpretation of the results.

    Focuses on the jsd -> volatility direction for both standard and controlled tests.
    """
    # Filter to JSD -> Volatility direction
    jsd_to_vol = granger_df[granger_df["direction"] == "jsd -> volatility"]
    
    # 1. Bivariate results
    biv_sig = jsd_to_vol[(~jsd_to_vol["controlled"]) & (jsd_to_vol["significant"])]
    
    # 2. Controlled results
    ctrl_sig = jsd_to_vol[(jsd_to_vol["controlled"]) & (jsd_to_vol["significant"])]

    lines = []
    lines.append("\n" + "=" * 70)
    lines.append("  Granger Causality - Interpretation Report")
    lines.append("=" * 70)

    # Bivariate analysis
    if not biv_sig.empty:
        best_biv = biv_sig.sort_values("p_value").iloc[0]
        lines.append(
            f"[OK] [Bivariate] Significant Granger causality found from JSD to Volatility!\n"
            f"  - Lowest p-value at lag {best_biv['lag']} week(s): F = {best_biv['f_stat']:.4f}, p = {best_biv['p_value']:.4f}.\n"
            f"  - Suggests rapid topic changes in Jio news precede abnormal RIL stock movement by ~{best_biv['lag']} week(s)."
        )
    else:
        # Check if there are borderline lags (0.05 <= p < 0.15) for honesty
        biv_borderline = jsd_to_vol[(~jsd_to_vol["controlled"]) & (jsd_to_vol["p_value"] < 0.15)]
        if not biv_borderline.empty:
            best_border = biv_borderline.sort_values("p_value").iloc[0]
            lines.append(
                f"[WARNING] [Bivariate] Marginal/weak evidence found from JSD to Volatility:\n"
                f"  - Lowest p-value at lag {best_border['lag']} week(s): F = {best_border['f_stat']:.4f}, p = {best_border['p_value']:.4f}.\n"
                f"  - Borderline p-value (between 0.05 and 0.15). Do not over-interpret."
            )
        else:
            lines.append(
                f"[NO] [Bivariate] No significant Granger causality found from JSD to Volatility at any lag 1-4 weeks.\n"
                f"  - This suggests narrative shift is coincident with or lag-free relative to price volatility,\n"
                f"    consistent with a semi-efficient market that processes media coverage rapidly."
            )

    lines.append("-" * 70)

    # Controlled analysis (VAR with NIFTY 50)
    if not ctrl_sig.empty:
        best_ctrl = ctrl_sig.sort_values("p_value").iloc[0]
        lines.append(
            f"[OK] [Controlled] Narrative shift adds significant predictive power beyond market context!\n"
            f"  - Controlling for NIFTY 50, lowest p-value at lag {best_ctrl['lag']} week(s): F = {best_ctrl['f_stat']:.4f}, p = {best_ctrl['p_value']:.4f}.\n"
            f"  - Confirms narrative shift has unique predictive signal for RIL stock volatility."
        )
    else:
        ctrl_borderline = jsd_to_vol[(jsd_to_vol["controlled"]) & (jsd_to_vol["p_value"] < 0.15)]
        if not ctrl_borderline.empty:
            best_border = ctrl_borderline.sort_values("p_value").iloc[0]
            lines.append(
                f"[WARNING] [Controlled] Marginal evidence when controlling for market returns:\n"
                f"  - Lowest p-value at lag {best_border['lag']} week(s): F = {best_border['f_stat']:.4f}, p = {best_border['p_value']:.4f}."
            )
        else:
            lines.append(
                f"[NO] [Controlled] No significant causality from JSD to Volatility when controlling for NIFTY 50.\n"
                f"  - Any apparent predictive relationship in the bivariate model disappears when controlling\n"
                f"    for broad market returns, indicating common exposure to market factors."
            )

    # Reverse direction analysis (Feedback loop)
    rev_sig = granger_df[(granger_df["direction"] == "volatility -> jsd") & (granger_df["significant"])]
    lines.append("-" * 70)
    if not rev_sig.empty:
        best_rev = rev_sig.sort_values("p_value").iloc[0]
        lines.append(
            f"[OK] [Reverse] Significant causality from Volatility to JSD found!\n"
            f"  - Lowest p-value at lag {best_rev['lag']} week(s): F = {best_rev['f_stat']:.4f}, p = {best_rev['p_value']:.4f}.\n"
            f"  - Suggests a feedback loop: price movement occurs first, and media topics shift in response."
        )
    else:
        lines.append(
            f"[NO] [Reverse] No significant feedback causality (Volatility -> JSD) found."
        )
        
    lines.append("=" * 70 + "\n")
    return "\n".join(lines)


def run_full_pipeline() -> pd.DataFrame:
    """Load inputs, execute tests, log summaries, save outputs, and return results."""
    # Verify input exists
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing aligned final dataset: {INPUT_PATH}")

    # Load dataset
    df = pd.read_parquet(INPUT_PATH)
    logger.info("Loaded aligned dataset: %d rows", len(df))

    # Run bivariate and controlled tests
    biv_results = run_granger_test(df, max_lag=4)
    ctrl_results = run_controlled_granger(df, max_lag=4)

    # Combine results
    combined_results = pd.concat([biv_results, ctrl_results], ignore_index=True)

    # Format output tables for printing
    print("\n" + "=" * 70)
    print("  GRANGER CAUSALITY TEST RESULTS")
    print("=" * 70)
    print(combined_results.to_string(index=False, formatters={
        "f_stat": lambda x: f"{x:.4f}",
        "p_value": lambda x: f"{x:.4f}",
    }))
    print("=" * 70 + "\n")

    # Log interpretations
    interpretation_str = interpret_results(combined_results)
    print(interpretation_str)

    # Save to parquet
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    combined_results.to_parquet(OUTPUT_PATH, index=False)
    logger.info("Saved granger results -> %s", OUTPUT_PATH)

    return combined_results


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_full_pipeline()
    sys.exit(0)
