from sqlalchemy import create_engine, text

# YOUR CORRECT PASSWORD - UPDATED
DB_PASSWORD = "asdasdasd"  # <--- I've set this to your password
DB_NAME = "market_db"
DB_USER = "postgres"
DB_HOST = "localhost"
DB_PORT = "5432"

# Create connection string
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Create engine
engine = create_engine(DATABASE_URL)

# Test the connection
try:
    with engine.connect() as conn:
        result = conn.execute(text("SELECT 1"))
        print("Database connection successful!")
        
        # Check if tables exist
        tables = conn.execute(text(
            """
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
            """
        )).fetchall()
        
        print(f"Tables in database: {[t[0] for t in tables]}")
        
except Exception as e:
    print(f"Connection failed: {e}")