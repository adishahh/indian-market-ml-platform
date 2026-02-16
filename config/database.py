import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker


class Database:
    def __init__(self):
        self.db_user = os.getenv("DB_USER", "postgres")
        self.db_password = os.getenv("DB_PASSWORD", "")
        self.db_host = os.getenv("DB_HOST", "localhost")
        self.db_port = os.getenv("DB_PORT", "5432")
        self.db_name = os.getenv("DB_NAME", "market_db")

        self.database_url = (
            f"postgresql://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )

        self.engine = create_engine(self.database_url, pool_pre_ping=True)
        self.SessionLocal = sessionmaker(bind=self.engine)

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
                """)
            )
            return [row[0] for row in result.fetchall()]
