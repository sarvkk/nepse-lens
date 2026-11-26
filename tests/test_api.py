from __future__ import annotations

from fastapi.testclient import TestClient

import nepse_lens.api as api


def test_health_and_catalog(trained, monkeypatch):
    db_path, _, _ = trained
    monkeypatch.setattr(api, "DB_PATH", db_path)
    client = TestClient(api.app)

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["tables"]["prices"] > 0

    tickers = client.get("/tickers")
    assert tickers.status_code == 200
    assert len(tickers.json()) > 0


def test_ohlcv_and_not_found(trained, monkeypatch):
    db_path, _, _ = trained
    monkeypatch.setattr(api, "DB_PATH", db_path)
    client = TestClient(api.app)

    response = client.get("/ohlcv/NABIL", params={"limit": 5})
    assert response.status_code == 200
    assert len(response.json()) == 5

    assert client.get("/ohlcv/NOPE").status_code == 404


def test_anomalies_sentiment_predictions(trained, monkeypatch):
    db_path, _, _ = trained
    monkeypatch.setattr(api, "DB_PATH", db_path)
    client = TestClient(api.app)

    anomalies = client.get("/anomalies", params={"limit": 5})
    assert anomalies.status_code == 200
    assert len(anomalies.json()) > 0

    sentiment = client.get("/sentiment", params={"limit": 10})
    assert sentiment.status_code == 200
    assert len(sentiment.json()) > 0

    predictions = client.get("/predictions", params={"limit": 5})
    assert predictions.status_code == 200
    assert len(predictions.json()) > 0


def test_market_summary(trained, monkeypatch):
    db_path, _, _ = trained
    monkeypatch.setattr(api, "DB_PATH", db_path)
    client = TestClient(api.app)

    response = client.get("/market/summary")
    assert response.status_code == 200
    body = response.json()
    assert body["symbols"] > 0
    assert 0.0 <= body["breadth"] <= 1.0
    assert len(body["top_gainers"]) == 5


def test_missing_database_returns_503(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "DB_PATH", tmp_path / "missing.duckdb")
    client = TestClient(api.app)
    assert client.get("/health").status_code == 503
