import pandas as pd
import yfinance as yf
import logging
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataIngestionError(Exception):
    """Custom exception for data ingestion failures."""
    pass

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    retry=retry_if_exception_type((Exception)),
    reraise=True
)
def fetch_market_data_with_retry(symbol: str, start_date: str = "2018-01-01", end_date: str = None) -> pd.DataFrame:
    """
    Fetches market data from Yahoo Finance with exponential backoff retry logic.
    """
    try:
        logger.info(f"Attempting to fetch data for {symbol}...")
        
        # yf.download often prints to stdout, so we might see double logs, which is fine.
        df = yf.download(
            symbol,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=True  # useful for getting adjusted close directly
        )

        if df.empty:
            logger.warning(f"Returned empty DataFrame for {symbol}")
            raise DataIngestionError(f"No data found for {symbol}")

        # Flatten multi-index columns if they exist (common in new yf versions)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]

        # Basic Validation
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
             raise DataIngestionError(f"Missing columns {missing_cols} for {symbol}")
        
        # Check for NaN in critical columns
        if df[['Open', 'Close']].isna().any().any():
             logger.warning(f"Found NaNs in price data for {symbol}. Dropping rows with NaNs.")
             df.dropna(subset=['Open', 'Close'], inplace=True)

        logger.info(f"Successfully fetched {len(df)} rows for {symbol}")
        return df

    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {str(e)}")
        raise

# run the function
if __name__ == "__main__":
    # Fetch Apple stock data
    df = fetch_market_data_with_retry("AAPL", start_date="2024-01-01", end_date="2024-02-01")
    print(df.head())  # Show first 5 rows
    print(f"\nShape: {df.shape}")  # Show dimensions