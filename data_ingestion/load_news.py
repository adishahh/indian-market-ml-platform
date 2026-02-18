import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from sqlalchemy import text
from config.database import engine
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# A subset of major stocks to fetch news for
STOCKS = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "BHARTIARTL.NS", "ITC.NS"]

def fetch_yfinance_news():
    news_data = []
    
    for symbol in STOCKS:
        logger.info(f"Fetching news for {symbol}...")
        try:
            ticker = yf.Ticker(symbol)
            news_items = ticker.news
            
            for item in news_items:
                # yfinance returns timestamp
                ts = item.get('providerPublishTime', 0)
                date = datetime.fromtimestamp(ts).date()
                title = item.get('title', '')
                publisher = item.get('publisher', 'Unknown')
                
                news_data.append({
                    "date": date,
                    "symbol": symbol,
                    "headline": title,
                    "source": publisher
                })
        except Exception as e:
            logger.error(f"Error for {symbol}: {e}")
            
    return pd.DataFrame(news_data)

def save_news(df):
    if df.empty:
        logger.warning("No news found.")
        return

    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS news (
                id SERIAL PRIMARY KEY,
                date DATE,
                symbol TEXT,
                headline TEXT,
                sentiment_score FLOAT,
                source TEXT,
                UNIQUE(headline, date) 
            )
        """))
        
        for row in df.itertuples(index=False):
            # Using ON CONFLICT to avoid duplicates
            conn.execute(text("""
                INSERT INTO news (date, symbol, headline, source)
                VALUES (:date, :symbol, :headline, :source)
                ON CONFLICT (headline, date) DO NOTHING
            """), {
                "date": row.date,
                "symbol": row.symbol,
                "headline": row.headline,
                "source": row.source
            })
    logger.info(f"Saved {len(df)} news items (skipping duplicates).")

if __name__ == "__main__":
    df = fetch_yfinance_news()
    save_news(df)
