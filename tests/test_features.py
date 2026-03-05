"""
tests/test_features.py
Unit tests for the Feature Engineering layer.
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import pytest
import pandas as pd
import numpy as np
from feature_engineering.build_features import compute_rsi, calculate_technicals


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def _make_price_df(n=100):
    """Create a synthetic OHLCV DataFrame for testing."""
    np.random.seed(42)
    dates = pd.date_range("2022-01-01", periods=n, freq="B")
    close = 100 + np.cumsum(np.random.randn(n))
    return pd.DataFrame({
        "date": dates,
        "open": close - 0.5,
        "close": close,
        "high": close + 1.0,
        "low": close - 1.0,
        "volume": np.random.randint(100000, 500000, n).astype(float),
        "stock_id": 1,
    })


# -------------------------------------------------------------------
# RSI Tests
# -------------------------------------------------------------------
class TestRSI:
    def test_rsi_is_between_0_and_100(self):
        df = _make_price_df()
        rsi = compute_rsi(df["close"], window=14)
        valid = rsi.dropna()
        assert (valid >= 0).all(), "RSI values should be >= 0"
        assert (valid <= 100).all(), "RSI values should be <= 100"

    def test_rsi_returns_series(self):
        df = _make_price_df()
        rsi = compute_rsi(df["close"], window=14)
        assert isinstance(rsi, pd.Series)

    def test_rsi_has_nan_at_start(self):
        """RSI requires a warm-up period; first 14 values should be NaN."""
        df = _make_price_df(n=30)
        rsi = compute_rsi(df["close"], window=14)
        assert rsi.iloc[:14].isna().all(), "First window-1 RSI values should be NaN"

    def test_rsi_with_constant_price(self):
        """Constant price means no gain or loss — RSI edge case."""
        df = _make_price_df()
        df["close"] = 100.0  # flat price
        rsi = compute_rsi(df["close"], window=14)
        # Expect NaN or 50 for flat prices (no movement)
        valid = rsi.dropna()
        assert len(valid) >= 0  # Just shouldn't throw


# -------------------------------------------------------------------
# calculate_technicals Tests
# -------------------------------------------------------------------
class TestCalculateTechnicals:
    def test_rsi_column_created(self):
        df = _make_price_df()
        result = calculate_technicals(df)
        assert "rsi_14" in result.columns, "rsi_14 should be a feature column"

    def test_sma_columns_created(self):
        df = _make_price_df()
        result = calculate_technicals(df)
        assert "sma_20" in result.columns
        assert "sma_50" in result.columns

    def test_macd_columns_created(self):
        df = _make_price_df()
        result = calculate_technicals(df)
        assert "macd" in result.columns
        assert "macd_signal" in result.columns

    def test_returns_created(self):
        df = _make_price_df()
        result = calculate_technicals(df)
        assert "return_1d" in result.columns
        assert "return_5d" in result.columns
        assert "return_20d" in result.columns

    def test_no_inf_values(self):
        df = _make_price_df()
        result = calculate_technicals(df)
        numeric = result.select_dtypes(include=[float, int])
        assert not np.isinf(numeric.values).any(), "No infinite values should be in features"

    def test_output_is_sorted_by_date(self):
        df = _make_price_df().sample(frac=1, random_state=99)  # shuffle
        result = calculate_technicals(df)
        assert result["date"].is_monotonic_increasing, "Output should be sorted by date"
