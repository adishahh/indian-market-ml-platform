    -- =========================================================
    -- Database Schema for Indian Market ML Platform
    -- =========================================================
    -- Purpose:
    -- 1. Store stock master data (NSE/BSE)
    -- 2. Store historical OHLCV price data
    -- 3. Store benchmark index data (e.g., NIFTY 50)
    -- =========================================================

    -- -----------------------------
    -- Drop tables (DEV ONLY)
    -- -----------------------------
    DROP TABLE IF EXISTS index_prices CASCADE;
    DROP TABLE IF EXISTS indices CASCADE;
    DROP TABLE IF EXISTS prices CASCADE;
    DROP TABLE IF EXISTS stocks CASCADE;

    -- -----------------------------
    -- Stocks Master Table
    -- -----------------------------
    CREATE TABLE stocks (
        stock_id SERIAL PRIMARY KEY,
        symbol TEXT UNIQUE NOT NULL,
        name TEXT,
        exchange TEXT DEFAULT 'NSE',
        sector TEXT,
        industry TEXT,
        is_active BOOLEAN DEFAULT TRUE,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    -- -----------------------------
    -- Stock Price History (OHLCV)
    -- -----------------------------
    CREATE TABLE prices (
        stock_id INT NOT NULL REFERENCES stocks(stock_id) ON DELETE CASCADE,
        date DATE NOT NULL,
        open FLOAT,
        high FLOAT,
        low FLOAT,
        close FLOAT,
        adjusted_close FLOAT,
        volume BIGINT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (stock_id, date)
    );

    -- Index for faster time-series queries
    CREATE INDEX idx_prices_stock_date
        ON prices(stock_id, date);

    -- -----------------------------
    -- Benchmark Indices Master
    -- -----------------------------
    CREATE TABLE indices (
        index_id SERIAL PRIMARY KEY,
        symbol TEXT UNIQUE NOT NULL,
        name TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    -- -----------------------------
    -- Index Price History
    -- -----------------------------
    CREATE TABLE index_prices (
        index_id INT NOT NULL REFERENCES indices(index_id) ON DELETE CASCADE,
        date DATE NOT NULL,
        close FLOAT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (index_id, date)
    );

    -- Index for faster benchmark queries
    CREATE INDEX idx_index_prices_index_date
        ON index_prices(index_id, date);

    -- =========================================================
    -- Verification Queries (Optional – run manually)
    -- =========================================================
    -- SELECT table_name
    -- FROM information_schema.tables
    -- WHERE table_schema = 'public'
    -- ORDER BY table_name;
