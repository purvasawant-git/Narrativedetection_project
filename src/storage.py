"""
storage.py  —  Phase 2: Persistent Storage  (Production v2)
NarrativeShiftProject / src/storage.py

Converts raw parquet batches → DuckDB with proper schema + indexes.
Supports incremental loading and efficient querying for downstream NLP phases.

Fixes vs v1
-----------
  - INSERT OR IGNORE via staging table  (v1 crashed on PRIMARY KEY conflict)
  - Accurate inserted-row count  (v1 used a 5-second timestamp race condition)
  - Column-order explicit INSERT  (v1 SELECT * from df with extra word_count col)
  - Timezone-aware published_at handled correctly before insert
"""

import logging
from pathlib import Path
from typing import Optional

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
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_DB_PATH = PROJECT_ROOT / "data" / "articles.duckdb"

DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
DATA_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# SQL
# ─────────────────────────────────────────────────────────────────────────────

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS articles (
    article_id    VARCHAR PRIMARY KEY,
    title         VARCHAR NOT NULL,
    content       VARCHAR,
    source        VARCHAR,
    published_at  TIMESTAMP,
    keyword_group VARCHAR,
    fetch_source  VARCHAR,
    word_count    INTEGER,
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

_CREATE_INDEX_DATE = """
CREATE INDEX IF NOT EXISTS idx_published_at
ON articles (published_at);
"""

_CREATE_INDEX_KW = """
CREATE INDEX IF NOT EXISTS idx_keyword_group
ON articles (keyword_group);
"""

# ─────────────────────────────────────────────────────────────────────────────
# Connection / schema
# ─────────────────────────────────────────────────────────────────────────────

def create_database(db_path: str = str(DATA_DB_PATH)) -> duckdb.DuckDBPyConnection:
    """
    Open (or create) the DuckDB database file and ensure the articles table
    and all indexes exist.  Safe to call multiple times (all DDL is idempotent).
    """
    conn = duckdb.connect(db_path)
    conn.execute(_CREATE_TABLE)
    conn.execute(_CREATE_INDEX_DATE)
    conn.execute(_CREATE_INDEX_KW)
    logger.info("DuckDB ready  —  %s", db_path)
    return conn


# ─────────────────────────────────────────────────────────────────────────────
# Insert
# ─────────────────────────────────────────────────────────────────────────────

def insert_articles(
    conn: duckdb.DuckDBPyConnection,
    df: pd.DataFrame,
) -> int:
    """
    Insert new articles, silently skipping rows whose article_id already exists.

    Strategy: load df into an in-memory staging table, then INSERT INTO articles
    SELECT … FROM staging WHERE article_id NOT IN (SELECT article_id FROM articles).
    This is safe, accurate, and avoids PRIMARY KEY conflict exceptions.

    Returns the number of newly inserted rows.
    """
    if df.empty:
        logger.warning("insert_articles: received empty DataFrame — nothing to do.")
        return 0

    # ── prepare dataframe ─────────────────────────────────────────────────────
    df = df.copy()

    required = [
        "article_id", "title", "content", "source",
        "published_at", "keyword_group", "fetch_source",
    ]
    for col in required:
        if col not in df.columns:
            df[col] = None

    # word_count from content
    df["word_count"] = (
        df["content"].fillna("").astype(str).str.split().str.len()
    )

    # Normalise published_at → naive UTC datetime (DuckDB TIMESTAMP has no tz)
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
    if pd.api.types.is_datetime64tz_dtype(df["published_at"]):
        df["published_at"] = df["published_at"].dt.tz_localize(None)

    # ── count before insert ───────────────────────────────────────────────────
    before: int = conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]

    # ── stage → deduplicated insert ───────────────────────────────────────────
    try:
        # Register df as a view DuckDB can query directly (zero-copy)
        conn.register("_staging", df)

        conn.execute("""
            INSERT INTO articles
                (article_id, title, content, source,
                 published_at, keyword_group, fetch_source, word_count)
            SELECT
                article_id, title, content, source,
                published_at, keyword_group, fetch_source, word_count
            FROM _staging
            WHERE article_id NOT IN (SELECT article_id FROM articles)
        """)

        conn.unregister("_staging")

    except Exception as exc:
        logger.error("insert_articles failed: %s", exc)
        return 0

    # ── accurate inserted count ───────────────────────────────────────────────
    after: int = conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
    inserted   = after - before

    logger.info(
        "Inserted %d new articles  (skipped %d duplicates)  |  DB total: %d",
        inserted, len(df) - inserted, after,
    )
    return inserted


# ─────────────────────────────────────────────────────────────────────────────
# Query helpers
# ─────────────────────────────────────────────────────────────────────────────

def fetch_articles(
    conn: duckdb.DuckDBPyConnection,
    start_date:    Optional[str] = None,
    end_date:      Optional[str] = None,
    keyword_group: Optional[str] = None,
    min_words:     Optional[int] = None,
) -> pd.DataFrame:
    """
    Fetch articles with optional filters.

    Parameters
    ----------
    start_date    : "YYYY-MM-DD"  inclusive
    end_date      : "YYYY-MM-DD"  inclusive
    keyword_group : exact match on keyword_group column
    min_words     : minimum word_count

    Returns
    -------
    pd.DataFrame sorted by published_at DESC, ready for NLP.
    """
    clauses: list[str] = ["1=1"]
    params:  list      = []

    if start_date:
        clauses.append("published_at >= ?")
        params.append(start_date)
    if end_date:
        clauses.append("published_at <= ?")
        params.append(end_date)
    if keyword_group:
        clauses.append("keyword_group = ?")
        params.append(keyword_group)
    if min_words is not None:
        clauses.append("word_count >= ?")
        params.append(min_words)

    sql = (
        "SELECT * FROM articles WHERE "
        + " AND ".join(clauses)
        + " ORDER BY published_at DESC"
    )

    try:
        out = conn.execute(sql, params).df()
        logger.info(
            "fetch_articles: %d rows  "
            "(start=%s  end=%s  group=%s  min_words=%s)",
            len(out), start_date, end_date, keyword_group, min_words,
        )
        return out
    except Exception as exc:
        logger.error("fetch_articles failed: %s", exc)
        return pd.DataFrame()


def get_weekly_counts(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Weekly article counts (week starts Monday, ISO convention)."""
    sql = """
        SELECT
            DATE_TRUNC('week', published_at) AS week_start,
            COUNT(*)                          AS article_count
        FROM articles
        WHERE published_at IS NOT NULL
        GROUP BY week_start
        ORDER BY week_start
    """
    try:
        df = conn.execute(sql).df()
        logger.info("get_weekly_counts: %d weeks", len(df))
        return df
    except Exception as exc:
        logger.error("get_weekly_counts failed: %s", exc)
        return pd.DataFrame()


def get_summary(conn: duckdb.DuckDBPyConnection) -> None:
    """Print a quick database summary to the log."""
    total = conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
    if total == 0:
        logger.info("Database is empty.")
        return

    row = conn.execute("""
        SELECT
            MIN(published_at) AS first_date,
            MAX(published_at) AS last_date,
            COUNT(DISTINCT keyword_group) AS n_groups,
            COUNT(DISTINCT source)        AS n_sources
        FROM articles
    """).fetchone()

    logger.info(
        "DB summary  |  total: %d  |  %s → %s  |  "
        "keyword groups: %d  |  sources: %d",
        total, row[0], row[1], row[2], row[3],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Parquet loader
# ─────────────────────────────────────────────────────────────────────────────

def load_all_parquets(data_dir: str = str(DATA_RAW_DIR)) -> pd.DataFrame:
    """
    Load and concatenate all *.parquet files from data/raw/.
    Returns a deduplicated DataFrame (safety net for cross-batch dupes).
    """
    files = sorted(Path(data_dir).glob("*.parquet"))
    if not files:
        logger.warning("load_all_parquets: no .parquet files in %s", data_dir)
        return pd.DataFrame()

    logger.info("Loading %d parquet files from %s …", len(files), data_dir)

    frames: list[pd.DataFrame] = []
    for i, pf in enumerate(files, 1):
        try:
            df = pd.read_parquet(pf, engine="pyarrow")
            logger.info("  [%d/%d]  %d rows  —  %s", i, len(files), len(df), pf.name)
            frames.append(df)
        except Exception as exc:
            logger.error("  [%d/%d]  Failed to load %s: %s", i, len(files), pf.name, exc)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    before   = len(combined)
    combined.drop_duplicates(subset=["article_id"], inplace=True)
    logger.info(
        "Loaded %d unique articles from %d files  (%d cross-file dupes removed)",
        len(combined), len(files), before - len(combined),
    )
    return combined


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info("=" * 64)
    logger.info("  Phase 2 — Parquet → DuckDB")
    logger.info("=" * 64)

    conn = create_database()
    df   = load_all_parquets()

    if not df.empty:
        inserted = insert_articles(conn, df)
        logger.info("Bulk load complete — %d new articles inserted.", inserted)
    else:
        logger.warning("No parquet data found — nothing inserted.")

    get_summary(conn)

    weekly = get_weekly_counts(conn)
    if not weekly.empty:
        logger.info(
            "Coverage  |  %d weeks  |  %s  →  %s",
            len(weekly),
            str(weekly["week_start"].min())[:10],
            str(weekly["week_start"].max())[:10],
        )

    conn.close()
    logger.info("Storage phase complete.")
