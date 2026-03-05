# 🇮🇳 Indian Market ML Platform

> Production-grade ML system for Indian equity market signal generation, portfolio evaluation, and automated retraining.

---

## 🚀 What This Does

1. **Ingests** daily OHLCV prices, macro indices (Nifty 50, BankNifty, Gold, Crude, USD/INR), and financial news for **61 Nifty stocks**
2. **Engineers** 21 technical features (RSI, SMA, MACD, Bollinger Bands, lag returns, sentiment)
3. **Trains** an XGBoost classifier to predict whether a stock will outperform the Nifty 50 index over the next 5 days
4. **Serves** real-time predictions via a FastAPI REST API
5. **Alerts** via Telegram when a Buy signal fires (probability > 60%)
6. **Monitors** open portfolio positions and fires Sell alerts on Take Profit (15%) or Stop Loss (-5%)
7. **Retrains** automatically every week — deploys only if the new model improves accuracy

---

## 🧱 Architecture

```
indian-market-ml-platform/
│
├── data_ingestion/           # Data collectors
│   ├── load_prices.py        # OHLCV from yfinance
│   ├── load_index.py         # Macro indices (Nifty, Gold, etc.)
│   ├── load_news.py          # NewsAPI (primary) + yfinance (fallback)
│   ├── load_stocks.py        # Stock universe seeder
│   └── market_data.py        # Retry-wrapped yfinance fetcher
│
├── feature_engineering/
│   ├── build_features.py     # RSI, SMA, MACD, lags, sentiment
│   ├── feature_store.py      # Pre-calculated features for API
│   └── sentiment_analysis.py # FinBERT scoring pipeline
│
├── model/
│   ├── prepare_dataset.py    # Joins features + macro + sentiment
│   ├── train_model.py        # XGBoost classifier with class balancing
│   ├── tune_model.py         # Optuna hyperparameter search
│   ├── walk_forward.py       # Walk-forward validation (no data leakage)
│   ├── backtest.py           # Portfolio backtest with real transaction costs
│   ├── evaluate.py           # Sharpe ratio + directional accuracy metrics
│   └── explain.py            # SHAP feature importance
│
├── api/
│   └── main.py               # FastAPI: /predict, /evaluate_positions, /retrain, /health
│
├── automation/
│   ├── retrain_pipeline.py   # Full auto-retrain: data → features → train → compare → deploy
│   └── n8n_workflow.json     # n8n workflow: 9:20 AM signals + sell monitoring
│
├── config/
│   ├── config.yaml           # All system settings (universe, model, backtest costs)
│   ├── best_params.yaml      # Optuna-tuned XGBoost hyperparameters
│   ├── database.py           # SQLAlchemy engine (env-variable driven)
│   ├── logger.py             # Centralized file + console logging
│   └── universe.yaml         # 61 active stock symbols
│
├── db/
│   └── schema.sql            # PostgreSQL table definitions
│
├── tests/
│   ├── test_features.py      # Unit tests for RSI, SMA, MACD calculations
│   ├── test_api.py           # API endpoint tests (/health, /predict, /evaluate_positions)
│   └── test_db.py            # DB connectivity test
│
├── Dockerfile                # API container
├── docker-compose.yml        # FastAPI + PostgreSQL one-command deploy
├── run_pipeline.py           # Full pipeline entrypoint (ingest → features → train → backtest)
└── requirements.txt
```

---

## ⚡ Quick Start (Local)

### 1. Install

```bash
python -m venv venv
.\venv\Scripts\activate           # Windows
pip install -r requirements.txt
```

### 2. Configure

Copy `.env.example` to `.env` and fill in:

```env
NEWSAPI_KEY=your_newsapi_key_here   # Get free key at newsapi.org/register
DB_PASSWORD=your_postgres_password  # Leave blank if using trusted auth
```

### 3. Run the full pipeline (first time)

```bash
python run_pipeline.py
```

This runs: data ingestion → feature engineering → model training → backtest.

### 4. Start the API

```bash
.\venv\Scripts\python.exe -m uvicorn api.main:app --host 0.0.0.0 --port 8003 --reload
```

API docs: [http://127.0.0.1:8003/docs](http://127.0.0.1:8003/docs)

---

## 🐳 Docker Deployment (One Command)

```bash
docker compose up -d
```

This starts:
- **FastAPI** on `http://localhost:8003`
- **PostgreSQL** on `localhost:5432`

Model artifacts and logs persist via Docker volumes.

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Check API + model status |
| `POST` | `/predict` | Get Buy/Sell signal for a stock |
| `POST` | `/evaluate_positions` | Check open portfolio positions for exits |
| `POST` | `/retrain` | Trigger automated retraining pipeline |

### Example: `/predict`

```bash
curl -X POST http://127.0.0.1:8003/predict \
  -H "Content-Type: application/json" \
  -d '{"symbol": "TCS"}'
```

```json
{
  "symbol": "TCS.NS",
  "prediction": "Buy",
  "probability": 0.74,
  "rsi": 48.3,
  "sentiment": 0.12,
  "date": "2026-03-05"
}
```

---

## 📊 Backtesting

The backtester applies **real Indian market transaction costs** on every trade:

| Cost | Rate |
|------|------|
| Brokerage | 0.03% per side |
| STT (Buy) | 0.10% |
| STT (Sell) | 0.10% |
| Slippage | 0.10% |
| **Total round-trip** | **~0.46%** |

Run multi-stock portfolio backtest:

```bash
python model/backtest.py
```

Results → `model/experiments/portfolio_backtest_results.csv`

---

## 🔁 Automated Weekly Retraining

The `/retrain` endpoint (or `automation/retrain_pipeline.py`) automatically:
1. Pulls fresh price + news data
2. Rebuilds feature store
3. Retrains XGBoost
4. Deploys new model **only if accuracy improves**

Configure n8n to call `POST http://127.0.0.1:8003/retrain` once a week.

---

## 🧪 Running Tests

```bash
.\venv\Scripts\python.exe -m pytest tests/ -v
```

---

## 📈 Model Details

| Property | Value |
|----------|-------|
| Algorithm | XGBoost Classifier |
| Target | Outperform Nifty 50 over next 5 days |
| Features | 21 (technical + macro + sentiment) |
| Universe | 61 Nifty stocks |
| Validation | Walk-forward (no data leakage) |
| Class balancing | Automatic `scale_pos_weight` |
| Accuracy | ~54% (beats random 50% baseline) |

---

## ⚙️ Configuration

All settings in `config/config.yaml`:

```yaml
backtest:
  initial_capital: 100000     # ₹1 Lakh
  brokerage_pct: 0.0003       # Realistic costs
  stt_buy_pct: 0.001
  slippage_pct: 0.001

automation:
  schedule: "Mon-Fri 09:20 AM"
  alert_threshold: 0.60       # Min probability to fire signal
```

---

## ⚠️ Disclaimer

This project is for **research and engineering demonstration purposes only**. It is not financial advice and is not intended for live trading without further due diligence.
