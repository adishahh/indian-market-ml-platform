# test_simple.py
import psycopg2

# Your database details
DB_NAME = "market_db"
DB_USER = "postgres"
DB_PASSWORD = "asdasdasd"  # Your password
DB_HOST = "localhost"
DB_PORT = "5432"

print("Attempting to connect to PostgreSQL...")

try:
    # Try to connect
    conn = psycopg2.connect(
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )
    
    print("CONNECTED SUCCESSFULLY!")
    
    # Create a cursor
    cur = conn.cursor()
    
    # Check what's in the database
    cur.execute("SELECT COUNT(*) FROM stocks")
    stock_count = cur.fetchone()[0]
    print(f"Number of stocks in database: {stock_count}")
    
    # Show all stocks
    cur.execute("SELECT symbol FROM stocks")
    stocks = cur.fetchall()
    if stocks:
        print("Stocks in your database:")
        for stock in stocks:
            print(f"   - {stock[0]}")
    
    # Close connection
    cur.close()
    conn.close()
    print("Connection closed")
    
except Exception as e:
    print(f"Connection failed: {e}")
    print("\n Troubleshooting tips:")
    print("1. Make sure PostgreSQL is running")
    print("2. Check if database 'market_db' exists")
    print("3. Verify password is correct: asdasdasd")
    print("4. Try connecting via pgAdmin to confirm")