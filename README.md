# Indian Market ML Platform

Production-ready ML system for Indian stock market analysis and prediction.

## Features

* Modular data ingestion (Indian equity market)
* Config-driven feature engineering
* XGBoost-based prediction engine
* News sentiment integration (NLP-ready)
* REST API for inference (FastAPI)
* Automated daily update & retraining pipeline
* PostgreSQL as single source of truth

## Project Structure

```
indian-market-ml-platform/
│
├── data_ingestion/        # Market & news data collectors
├── feature_engineering/  # Technical & sentiment features
├── models/               # Training, validation, inference
├── api/                  # FastAPI application
├── automation/           # Scheduled jobs (daily / weekly)
├── db/                   # Database-related logic
│   ├── schema.sql        # PostgreSQL schema
│   ├── database.py       # DB connection & session management
│   
├── config/               # Application-level configs
│   └── config.yaml
├── notebooks/            # EDA only (non-production)
├── tests/                # Unit & integration tests
│   └── test.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

## Configuration Files

### `config/config.yaml`

Controls market universe, model parameters, feature toggles, and automation frequency. Designed to make the system easily customizable without changing code.

### `db/database.yaml`

Stores database-related metadata such as table tracking, feature versioning, and model run identifiers. This allows reproducibility and auditability of experiments and predictions.

### `db/database.py`

Centralized PostgreSQL connection handling using SQLAlchemy. Ensures consistent database access across ingestion, training, API, and automation layers.

## Testing

Basic tests are included in the `tests/` directory to validate:

* Database connectivity
* Core data transformations
* Model input/output consistency

Testing is intentionally lightweight but structured to support future expansion.

## Design Philosophy

* Production-first, not notebook-first
* Time-aware validation (no data leakage)
* Clear separation of concerns (data, features, models, API)
* Extensible to other markets or asset classes

## Disclaimer

This project is for research and engineering demonstration purposes only. It is not intended as a live trading or investment advisory system.

---
license: mit
tags:
- finance
- indian-market
- algorithmic-trading
- time-series
- machine-learning
---

# Indian Market ML Trading Model

## Overview
This repository contains a machine learning model trained on Indian equity market data.
The model is designed for:
- Signal generation
- Risk-aware backtesting
- Strategy evaluation

## Features
- Market regime awareness
- Backtest metrics included
- JSON-based model artifacts for deployment

## Artifacts
- `artifacts/model.json` – trained model
- `artifacts/metrics.json` – evaluation metrics
- `artifacts/backtest_metrics.json` – backtest results

## Intended Use
Research, experimentation, and integration into automated trading pipelines.

## Disclaimer
This project is for research purposes only. Not financial advice.
