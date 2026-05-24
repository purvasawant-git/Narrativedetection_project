import duckdb
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DB_PATH = PROJECT_ROOT / "data" / "articles.duckdb"

conn = duckdb.connect(str(DATA_DB_PATH))

df = conn.execute("""
SELECT 
    COUNT(*) as total,
    SUM(CASE WHEN word_count < 50 THEN 1 ELSE 0 END) as short_articles,
    AVG(word_count) as avg_words
FROM articles
""").df()

print(df)