# Indian Market ML Pipeline Architecture 🏗️

This detailed architecture maps the full data lifecycle from external APIs to actionable trading signals.

## 1. High-Level Flow (Mermaid Diagram)
```mermaid
graph TD
    %% Styling
    classDef input fill:#f9f,stroke:#333,stroke-width:2px;
    classDef process fill:#bbf,stroke:#333,stroke-width:2px;
    classDef storage fill:#e1f7d5,stroke:#333,stroke-width:2px;
    classDef output fill:#ff9,stroke:#333,stroke-width:2px;

    %% 1. Input Sources
    subgraph "External Data"
        Yahoo[Yahoo Finance API]:::input
        NewsAPI[Google News / FinBERT]:::input
    end

    %% 2. Ingestion
    subgraph "Phase 1: Ingestion Layer"
        IngestScript[data/ingest_data.py]:::process
        Yahoo -->|OHLCV Data| IngestScript
        NewsAPI -->|Headlines| IngestScript
    end

    %% 3. Raw Storage
    subgraph "PostgreSQL Database (Raw)"
        PricesTable[(prices)]:::storage
        NewsTable[(news)]:::storage
        IngestScript -->|Store| PricesTable
        IngestScript -->|Store| NewsTable
    end

    %% 4. Feature Engineering
    subgraph "Phase 2 & 12: Feature Layer"
        BuildFeat[feature_engineering/build_features.py]:::process
        FeatStore[feature_engineering/feature_store.py]:::process
        
        PricesTable -->|Read| BuildFeat
        NewsTable -->|Read| BuildFeat
        
        BuildFeat -->|Calculate| FeaturesDaily[(features_daily)]:::storage
        FeatStore -->|Pre-compute & Merge| FeatureStoreTable[(feature_store)]:::storage
    end

    %% 5. Modeling
    subgraph "Phase 3: Model Layer"
        TrainScript[model/train_model.py]:::process
        HyperParams(config/best_params.yaml):::input
        ModelArtifact[[model_cls_YYYYMMDD.json]]:::output
        
        FeaturesDaily -->|Train Data| TrainScript
        HyperParams -->|Config| TrainScript
        TrainScript -->|Save| ModelArtifact
    end

    %% 6. Deployment & Automation
    subgraph "Phase 10 & 11: Production Layer"
        API[api/main.py]:::process
        N8N[n8n Workflow]:::process
        Telegram[Telegram Bot]:::output
        Logs[(prediction_logs)]:::storage
        
        FeatureStoreTable -->|Inputs| API
        ModelArtifact -->|Load| API
        
        N8N -->|Trigger 9:20 AM| API
        API -->|JSON Response| N8N
        API -->|Insert| Logs
        N8N -->|Alert| Telegram
    end
```

## 2. Component Breakdown

### **A. Ingestion Layer**
*   **Script:** `data/ingest_data.py`
*   **Role:** Fetches raw OHLCV (Open, High, Low, Close, Volume) data from Yahoo Finance and News Headlines.
*   **Frequency:** Daily at market close.
*   **Output:** Populates `prices` and `news` tables in PostgreSQL.

### **B. Feature Layer**
*   **Script:** `feature_engineering/build_features.py`
*   **Role:** Calculates technical indicators (RSI, SMA, MACD) and sentiment scores (FinBERT).
*   **Transformation:** Raw Prices -> Technical Features.
*   **Script:** `feature_engineering/feature_store.py` (Phase 12 Hardening)
*   **Role:** Creates a "Production-Ready" feature vector merging Macro, Technicals, and Sentiment.
*   **Output:** Populates `feature_store` table for fast API access.

### **C. Model Layer**
*   **Script:** `model/train_model.py`
*   **Role:** Trains an XGBoost Classifier on historical data.
*   **Input:** `features_daily` table.
*   **Config:** `config/best_params.yaml` (Optimized hyperparameters).
*   **Output:** Saves model artifact to `model/artifacts/model_cls_*.json`.

### **D. Production API**
*   **Script:** `api/main.py` (FastAPI)
*   **Role:** Serves predictions requests.
*   **Logic:** Loads latest `.json` model -> Reads `feature_store` -> Predicts -> Logs to DB.
*   **Endpoint:** `POST /predict`

### **E. Automation**
*   **Tool:** n8n (Workflow Automation)
*   **Role:** Orchestrator.
*   **Schedule:** Mon-Fri at 9:20 AM IST.
*   **Action:** Triggers API for all active stocks -> Filters for "Strong Buy" -> Sends Telegram Alert.

## 3. Data Dictionary
*   **`prices`**: Raw market data (Date, Open, close, Volume).
*   **`features_daily`**: Training dataset (Technicals + Targets).
*   **`feature_store`**: Inference dataset (Pre-merged, ready for API).
*   **`prediction_logs`**: Audit trail of every prediction made.
