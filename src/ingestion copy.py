"""
ingestion.py  —  Phase 1: Data Ingestion  (Final + Content Extraction)

Source   : GDELT DOC 2.0 API
Enhance  : newspaper3k full-article extraction
Output   : data/raw/YYYY-MM-DD_batch.parquet
"""

import hashlib
import logging
import random
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

from duckdb import query
import pandas as pd
import requests
from dotenv import load_dotenv

# NEW IMPORT
from newspaper import Article

# ─────────────────────────────────────────────────────────────────────────────
# Bootstrap
# ─────────────────────────────────────────────────────────────────────────────

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Keyword taxonomy
# ─────────────────────────────────────────────────────────────────────────────

COMPANY_TERMS: list[str] = [
    "Reliance Jio",
    "Jio Platforms",
    "Reliance Industries",
    "Mukesh Ambani",
]

SECTOR_TERMS: list[str] = [
    "5G India",
    "spectrum auction",
    "telecom tariff",
    "AGR dues",
    "Jio debt",
    "Jio subscribers",
    "Jio IPO",
    "Jio financial services",
]

KEYWORD_PAIRS: list[tuple[str, str]] = [
    (c, s) for c in COMPANY_TERMS for s in SECTOR_TERMS
]

# ─────────────────────────────────────────────────────────────────────────────
# Output schema
# ─────────────────────────────────────────────────────────────────────────────

COLUMNS: list[str] = [
    "article_id",
    "title",
    "content",
    "source",
    "published_at",
    "keyword_group",
    "fetch_source",
]

# ─────────────────────────────────────────────────────────────────────────────
# GDELT config
# ─────────────────────────────────────────────────────────────────────────────

GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
GDELT_MAX_RECORDS = 250
GDELT_TIMEOUT = 120
MAX_RETRIES = 3

QUERY_DELAY = (70.0, 90.0)
CHUNK_COOL = (180.0, 240.0)

RETRY_BACKOFF = [90.0, 150.0, 240.0]
RETRY_JITTER = (10.0, 25.0)

# ─────────────────────────────────────────────────────────────────────────────
# NEW: Newspaper3k extractor
# ─────────────────────────────────────────────────────────────────────────────

def extract_article(url: str) -> str:
    """
    Extract full article text using newspaper3k
    """
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text 
    except Exception:
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _article_id(title: str, published_at: str) -> str:
    return hashlib.sha256(f"{title}{published_at}".encode()).hexdigest()


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=COLUMNS)


def _coerce(df: pd.DataFrame) -> pd.DataFrame:
    for col in COLUMNS:
        if col not in df.columns:
            df[col] = None

    df["published_at"] = pd.to_datetime(
        df["published_at"], utc=True, errors="coerce"
    )

    return df[COLUMNS].copy()


def _gdelt_ts(date_str: str, eod: bool = False) -> str:
    dt = datetime.strptime(date_str, "%Y-%m-%d")

    if eod:
        dt = dt.replace(hour=23, minute=59, second=59)

    return dt.strftime("%Y%m%d%H%M%S")


def _month_ranges(start: str, end: str) -> list[tuple[str, str]]:
    out = []

    cur = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")

    while cur < end_dt:
        nxt = (
            cur.replace(month=cur.month + 1, day=1)
            if cur.month < 12
            else cur.replace(year=cur.year + 1, month=1, day=1)
        )

        out.append((
            cur.strftime("%Y-%m-%d"),
            min(nxt - timedelta(days=1), end_dt).strftime("%Y-%m-%d"),
        ))

        cur = nxt

    return out


def _sleep(lo: float, hi: float, reason: str) -> None:
    t = round(random.uniform(lo, hi), 1)
    logger.info("Waiting %.1fs  [%s]", t, reason)
    time.sleep(t)


# ─────────────────────────────────────────────────────────────────────────────
# GDELT Fetcher (UPDATED WITH CONTENT EXTRACTION)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_gdelt(query: str, from_date: str, to_date: str, q_idx=None, q_total=None) -> pd.DataFrame:

    params = {
        "query": query,
        "mode": "artlist",
        "format": "json",
        "maxrecords": GDELT_MAX_RECORDS,
        "startdatetime": _gdelt_ts(from_date),
        "enddatetime": _gdelt_ts(to_date, eod=True),
        "sort": "DateDesc",
    }

    for attempt in range(MAX_RETRIES):

        try:
            resp = requests.get(GDELT_URL, params=params, timeout=GDELT_TIMEOUT)
        except Exception:
            continue

        if resp.status_code != 200:
            continue

        try:
            payload = resp.json()
        except Exception:
            return _empty()

        articles = payload.get("articles") or []

        if not articles:
            return _empty()

        rows = []
        skipped = 0
        for i, a in enumerate(articles[:50]):   # limit for testing
            prefix = f"[{q_idx}/{q_total}]" if q_idx else ""
            url = a.get("url", "")
            print(f"{prefix} {i+1}/{len(articles)} articles → {query}")
            content = extract_article(url)

            if not content or len(content.split()) < 25:
                skipped += 1
                continue

            rows.append({
                "article_id": _article_id(
                    a.get("title") or "",
                    a.get("seendate", "")
                ),
                "title": a.get("title", ""),
                "content": content,
                "source": a.get("domain", ""),
                "published_at": a.get("seendate", ""),
                "keyword_group": query,
                "fetch_source": "gdelt",
            })
        logger.info(f"Skipped {skipped} low-quality articles for query: {query}")
        return _coerce(pd.DataFrame(rows))

    return _empty()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_ingestion(
    start: str = "2022-01-01",
    end: str = "2024-01-01",
    resume: bool = False,
) -> pd.DataFrame:

    chunks = _month_ranges(start, end)

    all_batches = []

    for chunk_idx, (cs, ce) in enumerate(chunks, 1):

        parquet_path = DATA_RAW_DIR / f"{cs}_batch.parquet"

        frames = []

        for q_idx, (company, sector) in enumerate(KEYWORD_PAIRS, 1):

            query = f"{company} + {sector}"
            print(f"\nQuery: {query} | {cs} → {ce}")    

            df = fetch_gdelt(
                query,
                cs,
                ce,
                q_idx=q_idx,
                q_total=len(KEYWORD_PAIRS)
            )

            if not df.empty:
                frames.append(df)

            _sleep(*QUERY_DELAY, "query delay")

        if frames:

            batch = pd.concat(frames, ignore_index=True)

            # Merge with existing parquet if it exists
            if parquet_path.exists():
                existing = pd.read_parquet(parquet_path, engine="pyarrow")
                batch = pd.concat([existing, batch], ignore_index=True)
                logger.info("Merged with existing batch: %s", parquet_path.name)

            batch.drop_duplicates(subset=["article_id"], inplace=True)

            batch.to_parquet(parquet_path, engine="pyarrow", index=False)

            all_batches.append(batch)

        if chunk_idx < len(chunks):
            _sleep(*CHUNK_COOL, "chunk cooldown")

    if not all_batches:
        return _empty()

    full = pd.concat(all_batches, ignore_index=True)

    full.drop_duplicates(subset=["article_id"], inplace=True)

    return full


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    _start = sys.argv[1] if len(sys.argv) > 1 else "2022-01-01"
    _end = sys.argv[2] if len(sys.argv) > 2 else "2024-01-01"

    result = run_ingestion(start=_start, end=_end)

    print(f"\nTotal unique articles collected: {len(result)}")