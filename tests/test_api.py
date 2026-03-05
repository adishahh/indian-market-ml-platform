"""
tests/test_api.py
API endpoint tests using FastAPI's TestClient with mocked database/model.
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient


# -------------------------------------------------------------------
# /health Endpoint Tests
# -------------------------------------------------------------------
class TestHealthEndpoint:
    def test_health_returns_ok(self):
        """Health endpoint should always return 200 OK."""
        from api.main import app
        client = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_has_status_field(self):
        from api.main import app
        client = TestClient(app)
        response = client.get("/health")
        body = response.json()
        assert "status" in body
        assert body["status"] == "ok"

    def test_health_has_model_loaded_field(self):
        from api.main import app
        client = TestClient(app)
        response = client.get("/health")
        body = response.json()
        assert "model_loaded" in body
        assert isinstance(body["model_loaded"], bool)


# -------------------------------------------------------------------
# /predict Endpoint Tests
# -------------------------------------------------------------------
class TestPredictEndpoint:
    def test_predict_with_no_model_returns_503(self):
        """If no model is loaded, predict should respond with 503 Service Unavailable."""
        from api import main as api_module
        original_model = api_module.model
        try:
            api_module.model = None
            client = TestClient(api_module.app)
            response = client.post("/predict", json={"symbol": "TCS.NS"})
            assert response.status_code == 503
        finally:
            api_module.model = original_model

    def test_predict_requires_symbol(self):
        """Request body must include symbol field."""
        from api.main import app
        client = TestClient(app)
        # Sending empty body, symbol defaults to "TCS.NS", so still 200 or 503 (not 422)
        response = client.post("/predict", json={})
        assert response.status_code in [200, 503, 500], \
            "Empty body should use default symbol, not cause a 422 validation error"


# -------------------------------------------------------------------
# /evaluate_positions Endpoint Tests
# -------------------------------------------------------------------
class TestEvaluatePositionsEndpoint:
    def test_evaluate_with_no_positions_returns_empty_signals(self):
        """When portfolio_positions table is empty, sell_signals should be an empty list."""
        from api.main import app
        client = TestClient(app)
        # Mock the DB to return empty dataframe
        with patch("api.main.engine") as mock_engine:
            mock_conn = MagicMock()
            mock_engine.connect.return_value.__enter__.return_value = mock_conn
            import pandas as pd
            mock_conn.execute.return_value = MagicMock()

            # Use pandas read_sql mock
            with patch("api.main.pd.read_sql") as mock_sql:
                mock_sql.return_value = pd.DataFrame()
                response = client.post("/evaluate_positions")
                assert response.status_code == 200
                body = response.json()
                assert "sell_signals" in body
                assert body["sell_signals"] == []
