import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# =========================
# Environment variables
# =========================
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "asdasdasd")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "market_db")

DATABASE_URL = (
    f"postgresql://{DB_USER}:{DB_PASSWORD}"
    f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

# =========================
# GLOBAL ENGINE (IMPORTANT)
# =========================
engine = create_engine(DATABASE_URL, pool_pre_ping=True)

SessionLocal = sessionmaker(bind=engine)


# =========================
# OPTIONAL DATABASE CLASS
# =========================
class Database:
    def __init__(self):
        self.engine = engine
        self.SessionLocal = SessionLocal

    def get_engine(self):
        return self.engine

    def get_session(self):
        return self.SessionLocal()

    def test_connection(self):
        with self.engine.connect() as conn:
            conn.execute(text("SELECT 1"))

    def list_tables(self):
        with self.engine.connect() as conn:
            result = conn.execute(
                text("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
                ORDER BY table_name
                """)
            )
            return [row[0] for row in result.fetchall()]
