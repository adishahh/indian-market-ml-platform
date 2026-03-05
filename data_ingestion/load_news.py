"""
data_ingestion/load_news.py

News ingestion with dual-source support:
  - Primary: NewsAPI (newsapi.org) — comprehensive real-time news
  - Fallback: yfinance ticker news (free, limited)

Setup:
  Set environment variable: $env:NEWSAPI_KEY="your_newsapi_key_here"
  Get a free key at: https://newsapi.org/register
"""
import os
import sys
sys.path.append(os.getcwd())

import requests
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from sqlalchemy import text
from config.database import engine
from config.logger import get_logger

logger = get_logger(__name__)

# Stocks tracked for news
STOCKS = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS",
    "ICICIBANK.NS", "SBIN.NS", "BHARTIARTL.NS", "ITC.NS"
]
STOCK_MAP = {s.replace(".NS", ""): s for s in STOCKS}

# -------------------------------------------------------
# FinBERT Sentiment Scoring
# -------------------------------------------------------
_sentiment_pipeline = None

def get_sentiment_pipeline():
    global _sentiment_pipeline
    if _sentiment_pipeline is None:
        try:
            from transformers import pipeline
            logger.info("Loading FinBERT model (ProsusAI/finbert)...")
            _sentiment_pipeline = pipeline(
                "text-classification",
                model="ProsusAI/finbert",
                tokenizer="ProsusAI/finbert",
                truncation=True,
                max_length=512
            )
            logger.info("FinBERT loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load FinBERT: {e}")
            _sentiment_pipeline = None
    return _sentiment_pipeline


def score_sentiment(headlines: list) -> list:
    pipe = get_sentiment_pipeline()
    if pipe is None:
        return [None] * len(headlines)

    label_map = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
    scores = []
    try:
        results = pipe(headlines, batch_size=16)
        for r in results:
            scores.append(label_map.get(r["label"].lower(), 0.0))
    except Exception as e:
        logger.error(f"FinBERT inference error: {e}")
        scores = [None] * len(headlines)
    return scores


# -------------------------------------------------------
# Primary Source: NewsAPI
# -------------------------------------------------------
def fetch_newsapi_news(api_key: str) -> pd.DataFrame:
    """
    Fetches Indian market news from NewsAPI.
    Free plan allows fetching last 30 days of news.
    """
    logger.info("Fetching news from NewsAPI (primary source)...")
    news_data = []

    # Build query combining all stock symbols
    symbols_query = " OR ".join(STOCK_MAP.keys())
    query = f"({symbols_query}) AND (stock OR market OR NSE OR BSE)"
    from_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "from": from_date,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 100,
        "apiKey": api_key,
    }

    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        articles = resp.json().get("articles", [])
        logger.info(f"NewsAPI returned {len(articles)} articles.")

        today = datetime.now().date()
        for article in articles:
            headline = article.get("title", "").strip()
            if not headline or headline.lower() == "[removed]":
                continue

            pub_at = article.get("publishedAt", "")
            try:
                date = datetime.fromisoformat(pub_at.replace("Z", "+00:00")).date()
            except Exception:
                date = today

            source = article.get("source", {}).get("name", "NewsAPI")

            # Try to match to a known stock symbol
            matched_symbol = "GENERAL"
            for sym in STOCK_MAP.keys():
                if sym.upper() in headline.upper():
                    matched_symbol = STOCK_MAP[sym]
                    break

            news_data.append({
                "date": date,
                "symbol": matched_symbol,
                "headline": headline,
                "source": source,
            })

    except Exception as e:
        logger.error(f"NewsAPI request failed: {e}")

    return pd.DataFrame(news_data)


# -------------------------------------------------------
# Fallback Source: yfinance
# -------------------------------------------------------
def fetch_yfinance_news() -> pd.DataFrame:
    """Fallback: fetch news from yfinance (free, limited coverage)."""
    logger.info("Fetching news from yfinance (fallback source)...")
    news_data = []

    for symbol in STOCKS:
        logger.info(f"  yfinance: fetching news for {symbol}...")
        try:
            ticker = yf.Ticker(symbol)
            news_items = ticker.news

            for item in news_items:
                content = item.get("content", item)
                title = content.get("title", "") or item.get("title", "")
                if not title:
                    continue

                pub_date = content.get("pubDate", "")
                if pub_date:
                    try:
                        date = datetime.fromisoformat(pub_date.replace("Z", "+00:00")).date()
                    except Exception:
                        date = datetime.utcnow().date()
                else:
                    ts = item.get("providerPublishTime", 0)
                    date = datetime.fromtimestamp(ts).date() if ts else datetime.utcnow().date()

                provider = content.get("provider", {})
                publisher = provider.get("displayName", "") if isinstance(provider, dict) else ""
                if not publisher:
                    publisher = item.get("publisher", "yfinance")

                news_data.append({
                    "date": date,
                    "symbol": symbol,
                    "headline": title,
                    "source": publisher,
                })

        except Exception as e:
            logger.error(f"yfinance news error for {symbol}: {e}")

    return pd.DataFrame(news_data)


# -------------------------------------------------------
# Main Entrypoint
# -------------------------------------------------------
def save_news(df: pd.DataFrame):
    if df.empty:
        logger.warning("No news found — nothing to save.")
        return

    logger.info(f"Scoring {len(df)} headlines with FinBERT...")
    df["sentiment_score"] = score_sentiment(df["headline"].tolist())
    logger.info("Sentiment scoring complete.")

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

        saved = 0
        for row in df.itertuples(index=False):
            sentiment = getattr(row, "sentiment_score", None)
            try:
                conn.execute(text("""
                    INSERT INTO news (date, symbol, headline, sentiment_score, source)
                    VALUES (:date, :symbol, :headline, :sentiment_score, :source)
                    ON CONFLICT (headline, date)
                    DO UPDATE SET sentiment_score = EXCLUDED.sentiment_score
                """), {
                    "date": row.date,
                    "symbol": row.symbol,
                    "headline": row.headline,
                    "sentiment_score": float(sentiment) if sentiment is not None else None,
                    "source": row.source,
                })
                saved += 1
            except Exception as e:
                logger.warning(f"Skipping duplicate or errored row: {e}")

    logger.info(f"Saved/updated {saved} news items in DB.")


def load_news():
    """
    Main news ingestion function.
    Tries NewsAPI first, falls back to yfinance if unavailable.
    """
    newsapi_key = os.environ.get("NEWSAPI_KEY", "").strip()

    if newsapi_key:
        df = fetch_newsapi_news(newsapi_key)
        if df.empty:
            logger.warning("NewsAPI returned empty results — switching to yfinance fallback.")
            df = fetch_yfinance_news()
    else:
        logger.warning(
            "NEWSAPI_KEY environment variable not set. "
            "Falling back to yfinance news. "
            "Set $env:NEWSAPI_KEY='your_key' for better coverage."
        )
        df = fetch_yfinance_news()

    save_news(df)


if __name__ == "__main__":
    load_news()
