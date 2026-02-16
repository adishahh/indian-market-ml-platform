# scripts/debug_db.py
import os
import psycopg2

DB_NAME = os.getenv("DB_NAME", "market_db")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")

def main():
    print("Attempting to connect to PostgreSQL...")

    try:
        conn = psycopg2.connect(
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )

        print("CONNECTED SUCCESSFULLY!")

        cur = conn.cursor()

        cur.execute("SELECT COUNT(*) FROM stocks")
        stock_count = cur.fetchone()[0]
        print(f"Number of stocks in database: {stock_count}")

        cur.execute("SELECT symbol FROM stocks")
        stocks = cur.fetchall()

        if stocks:
            print("Stocks in your database:")
            for stock in stocks:
                print(f"   - {stock[0]}")

        cur.close()
        conn.close()
        print("Connection closed")

    except Exception as e:
        print(f"Connection failed: {e}")

if __name__ == "__main__":
    main()
