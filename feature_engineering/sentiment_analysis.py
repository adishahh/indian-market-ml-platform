from transformers import pipeline
import pandas as pd
from sqlalchemy import text
from config.database import engine
import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Use GPU if available
device = 0 if torch.cuda.is_available() else -1

def analyze_sentiment():
    logger.info("Loading FinBERT from Hugging Face...")
    try:
        pipe = pipeline("text-classification", model="ProsusAI/finbert", device=device)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return

    # Fetch news without scores
    query = "SELECT id, headline FROM news WHERE sentiment_score IS NULL"
    with engine.connect() as conn:
        df = pd.read_sql(text(query), conn)
    
    if df.empty:
        logger.info("No new news to analyze.")
        return

    logger.info(f"Analyzing {len(df)} headlines...")
    
    # Batch processing
    batch_size = 32
    updates = []
    
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size]
        try:
            results = pipe(batch['headline'].tolist())
            
            for idx, res in zip(batch['id'], results):
                # FinBERT returns labels: positive, negative, neutral
                # We map to score: +1, -1, 0
                score = 0
                if res['label'] == 'positive': score = res['score']
                elif res['label'] == 'negative': score = -res['score']
                else: score = 0
                
                updates.append({"id": idx, "score": score})
        except Exception as e:
            logger.error(f"Error processing batch {i}: {e}")
            
    # Save back to DB
    if updates:
        logger.info(f"Updating database with {len(updates)} scores...")
        with engine.begin() as conn:
            for up in updates:
                conn.execute(text("""
                    UPDATE news SET sentiment_score = :score WHERE id = :id
                """), {"score": up['score'], "id": up['id']})
            
    logger.info("Sentiment analysis complete.")

if __name__ == "__main__":
    analyze_sentiment()
