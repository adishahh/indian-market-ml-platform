-- ============================================
-- Database Schema for Indian Market ML Platform
-- ============================================

-- Drop tables if they exist (for clean reinstallation)
DROP TABLE IF EXISTS prices CASCADE;
DROP TABLE IF EXISTS stocks CASCADE;

-- Create stocks table
CREATE TABLE stocks (
    stock_id SERIAL PRIMARY KEY,
    symbol TEXT UNIQUE NOT NULL
);

-- Create prices table
CREATE TABLE prices (
    stock_id INT REFERENCES stocks(stock_id),
    date DATE NOT NULL,
    open FLOAT,
    close FLOAT,
    volume BIGINT,
    PRIMARY KEY (stock_id, date)
);

-- Create index for faster queries
CREATE INDEX idx_prices_stock_date ON prices(stock_id, date);

-- ============================================
-- Optional: Insert some sample Indian stocks
-- ============================================
INSERT INTO stocks (symbol) VALUES 
    ('RELIANCE'), 
    ('TCS'), 
    ('HDFCBANK'),
    ('INFY'),
    ('ICICIBANK'),
    ('SBIN'),
    ('BHARTIARTL'),
    ('ITC'),
    ('WIPRO'),
    ('AXISBANK')
ON CONFLICT (symbol) DO NOTHING;

-- ============================================
-- Verify the setup
-- ============================================
-- Check stocks
SELECT 'Stocks count: ' || COUNT(*) FROM stocks;

-- Check tables exist
SELECT table_name 
FROM information_schema.tables 
WHERE table_schema = 'public'
ORDER BY table_name;