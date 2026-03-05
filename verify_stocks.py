import sys
sys.path.append('.')
from config.database import engine
from sqlalchemy import text

with engine.connect() as conn:
    rows = conn.execute(text("""
        SELECT s.symbol, COUNT(p.stock_id) as price_rows
        FROM stocks s
        LEFT JOIN prices p ON s.stock_id = p.stock_id
        GROUP BY s.symbol
        ORDER BY s.symbol
    """)).fetchall()

print(f"{'Symbol':<15} {'Price Rows':>12}")
print("-" * 30)
for sym, cnt in rows:
    status = "✅" if cnt > 0 else "❌ NO DATA"
    print(f"{sym:<15} {cnt:>10}  {status}")
