"""
finance.py — Phase 7: Financial Data Pipeline
==============================================
Downloads daily price data for RELIANCE.NS (RIL) and NIFTY 50 (^NSEI) from
Yahoo Finance, resamples to ISO-week granularity (Monday-anchored), computes
weekly returns and volatility, and saves the result to data/financial_metrics.parquet.

CRITICAL ALIGNMENT:
    week_start is always the Monday of the ISO week (weekday == 0).
    This must match preprocessing.py's week_start convention so the
    Phase 8 merge produces no NaN rows.
"""

import logging
import pathlib
import sys

import numpy as np
import pandas as pd
# pyrefly: ignore [missing-import]
import yfinance as yf

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
# 1. Download raw daily price data
# ---------------------------------------------------------------------------

def download_stock_data(
    ticker: str = "RELIANCE.NS",
    start: str = "2022-01-01",
    end: str = "2024-01-01",
) -> pd.DataFrame:
    """Download daily OHLCV data for *ticker* from Yahoo Finance.

    Parameters
    ----------
    ticker : str
        Yahoo Finance ticker symbol (e.g. "RELIANCE.NS", "^NSEI").
    start  : str  ISO date string, inclusive.
    end    : str  ISO date string, exclusive.

    Returns
    -------
    pd.DataFrame
        Daily OHLCV dataframe with a DatetimeIndex (UTC-naive).

    Raises
    ------
    ValueError
        If the download returns an empty dataframe.
    """
    logger.info("Downloading %s from %s to %s …", ticker, start, end)
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)

    if df.empty:
        raise ValueError(
            f"yfinance returned no data for ticker '{ticker}' "
            f"between {start} and {end}. "
            "Check the ticker symbol and your internet connection."
        )

    # Flatten MultiIndex columns that yfinance sometimes produces
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Ensure the index is a proper DatetimeIndex
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"

    logger.info(
        "Downloaded %d trading days for %s (%s → %s).",
        len(df),
        ticker,
        df.index.min().date(),
        df.index.max().date(),
    )
    return df


# ---------------------------------------------------------------------------
# 2. Compute weekly metrics from daily data
# ---------------------------------------------------------------------------

def _weekly_return(close: pd.Series) -> float:
    """Return (last_close / first_close) - 1 for a single week's close series."""
    if len(close) < 2:
        return np.nan
    return float(close.iloc[-1] / close.iloc[0]) - 1.0


def _weekly_vol(close: pd.Series) -> float:
    """Std of daily log returns within a single week."""
    if len(close) < 2:
        return np.nan
    log_rets = np.log(close / close.shift(1)).dropna()
    return float(log_rets.std())


def compute_weekly_metrics(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Group daily OHLCV to ISO-week (Monday-anchored) metrics.

    Computes:
        weekly_return     — (last close / first close) − 1
        weekly_volatility — std of within-week daily log returns
        rolling_vol_4w    — 4-week rolling mean of weekly_volatility

    The *week_start* column is always the Monday of the ISO week so it aligns
    with the news preprocessing.py week convention.

    Parameters
    ----------
    daily_df : pd.DataFrame
        Output of :func:`download_stock_data`.

    Returns
    -------
    pd.DataFrame
        Columns: week_start (datetime), weekly_return, weekly_volatility,
        rolling_vol_4w.
    """
    # Ensure index is sorted chronologically
    df_sorted = daily_df.sort_index()

    # Calculate Monday of the week for each daily data point
    df_sorted["week_start"] = df_sorted.index - pd.to_timedelta(df_sorted.index.weekday, unit="D")

    # Group by week_start
    grouped = df_sorted.groupby("week_start")["Close"]

    weekly_return = grouped.apply(_weekly_return)
    weekly_vol = grouped.apply(_weekly_vol)

    metrics = pd.DataFrame(
        {
            "week_start": weekly_return.index,
            "weekly_return": weekly_return.values,
            "weekly_volatility": weekly_vol.values,
        }
    )

    # Drop weeks with no trading data (e.g. pure holiday weeks)
    metrics = metrics.dropna(subset=["weekly_return", "weekly_volatility"]).copy()

    # 4-week rolling average volatility (min 2 periods to avoid leading NaNs)
    metrics["rolling_vol_4w"] = (
        metrics["weekly_volatility"].rolling(window=4, min_periods=2).mean()
    )

    # Reset index so week_start is a plain column (not the index)
    metrics = metrics.reset_index(drop=True)

    # Sanity check: all week_start values must be Mondays (weekday == 0)
    bad = metrics[metrics["week_start"].dt.weekday != 0]
    if not bad.empty:
        logger.warning(
            "Found %d week_start values that are NOT Mondays — check grouping logic!",
            len(bad),
        )
    else:
        logger.info(
            "All %d week_start values confirmed as Mondays ✓", len(metrics)
        )

    logger.info(
        "Weekly metrics computed: %d weeks, mean_vol=%.4f, max_drawdown=%.4f",
        len(metrics),
        metrics["weekly_volatility"].mean(),
        metrics["weekly_return"].min(),
    )
    return metrics


# ---------------------------------------------------------------------------
# 3. Market context — NIFTY 50 weekly return (control variable for Phase 9)
# ---------------------------------------------------------------------------

def get_market_context(
    start: str = "2022-01-01",
    end: str = "2024-01-01",
) -> pd.DataFrame:
    """Download NIFTY 50 and compute weekly returns as a market control variable.

    Parameters
    ----------
    start, end : str  ISO date strings.

    Returns
    -------
    pd.DataFrame
        Columns: week_start (datetime), nifty_return (float).
    """
    nifty_daily = download_stock_data("^NSEI", start=start, end=end)
    df_sorted = nifty_daily.sort_index()

    # Calculate Monday of the week for each daily data point
    df_sorted["week_start"] = df_sorted.index - pd.to_timedelta(df_sorted.index.weekday, unit="D")

    nifty_weekly = df_sorted.groupby("week_start")["Close"].apply(_weekly_return)

    ctx = pd.DataFrame(
        {
            "week_start": nifty_weekly.index,
            "nifty_return": nifty_weekly.values,
        }
    ).dropna(subset=["nifty_return"]).reset_index(drop=True)

    logger.info(
        "NIFTY 50 market context: %d weeks, mean_return=%.4f",
        len(ctx),
        ctx["nifty_return"].mean(),
    )
    return ctx


# ---------------------------------------------------------------------------
# 4. Build the combined financial dataset and save to parquet
# ---------------------------------------------------------------------------

def build_financial_dataset(
    ticker: str = "RELIANCE.NS",
    start: str = "2022-01-01",
    end: str = "2024-01-01",
    output_path: pathlib.Path | None = None,
) -> pd.DataFrame:
    """End-to-end: download, compute metrics, merge with market context, save.

    Parameters
    ----------
    ticker      : str   Stock ticker (default RELIANCE.NS).
    start, end  : str   Date range.
    output_path : Path  Where to save the parquet file.
                        Defaults to data/financial_metrics.parquet.

    Returns
    -------
    pd.DataFrame
        Final merged weekly financial metrics.
    """
    if output_path is None:
        output_path = DATA_DIR / "financial_metrics.parquet"

    # --- RIL data -----------------------------------------------------------
    ril_daily = download_stock_data(ticker, start=start, end=end)
    ril_weekly = compute_weekly_metrics(ril_daily)

    # --- NIFTY 50 control variable -----------------------------------------
    nifty_ctx = get_market_context(start=start, end=end)

    # --- Merge on week_start ------------------------------------------------
    # Ensure both are datetime for a clean merge
    ril_weekly["week_start"] = pd.to_datetime(ril_weekly["week_start"])
    nifty_ctx["week_start"] = pd.to_datetime(nifty_ctx["week_start"])

    merged = pd.merge(ril_weekly, nifty_ctx, on="week_start", how="left")

    # Final sort
    merged = merged.sort_values("week_start").reset_index(drop=True)

    # --- Save ---------------------------------------------------------------
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(output_path, index=False)
    logger.info("Saved financial metrics → %s", output_path)

    # --- Summary ------------------------------------------------------------
    max_drawdown_week = merged.loc[merged["weekly_return"].idxmin(), "week_start"]
    print("\n" + "=" * 60)
    print("  Financial Dataset Summary")
    print("=" * 60)
    print(f"  Date range   : {merged['week_start'].min().date()} → {merged['week_start'].max().date()}")
    print(f"  Total weeks  : {len(merged)}")
    print(f"  Mean weekly vol : {merged['weekly_volatility'].mean():.4f}")
    print(f"  Max weekly drawdown : {merged['weekly_return'].min():.4f}  (week of {max_drawdown_week.date()})")
    print(f"  Mean NIFTY return   : {merged['nifty_return'].mean():.4f}")
    print(f"  Saved to     : {output_path}")
    print("=" * 60 + "\n")

    return merged


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Phase 7 — Download RIL + NIFTY 50 weekly financial metrics."
    )
    parser.add_argument("--ticker", default="RELIANCE.NS", help="Stock ticker (default: RELIANCE.NS)")
    parser.add_argument("--start", default="2022-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default="2024-01-01", help="End date YYYY-MM-DD")
    args = parser.parse_args()

    df = build_financial_dataset(ticker=args.ticker, start=args.start, end=args.end)
    print(df.head(10).to_string(index=False))
    sys.exit(0)
