from config.database import engine
from sqlalchemy import text

with engine.connect() as conn:
    result = conn.execute(
        text("SELECT table_name FROM information_schema.tables WHERE table_schema='public'")
    )
    print("Tables:", [row[0] for row in result.fetchall()])
