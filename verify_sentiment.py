import sys
sys.path.append('.')
from config.database import engine
from sqlalchemy import text

with engine.connect() as conn:
    total = conn.execute(text("SELECT COUNT(*) FROM news")).fetchone()[0]
    scored = conn.execute(text("SELECT COUNT(*) FROM news WHERE sentiment_score IS NOT NULL")).fetchone()[0]
    rows = conn.execute(text(
        "SELECT symbol, headline, sentiment_score FROM news WHERE sentiment_score IS NOT NULL ORDER BY id DESC LIMIT 6"
    )).fetchall()

print(f"Total news rows: {total}")
print(f"Rows with sentiment_score: {scored}")
print()
for sym, h, s in rows:
    label = "POS" if s > 0 else ("NEG" if s < 0 else "NEU")
    print(f"[{label}] [{sym}] {h[:65]}")
