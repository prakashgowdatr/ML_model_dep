"""
tests/test_api.py — Tests for the FastAPI application.

Uses FastAPI's TestClient (synchronous HTTP) and mocks the model
so tests run instantly without loading real artifacts.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def client():
    """
    Create a TestClient with the model loading mocked out.

    We patch:
      1. load_artifacts() — so the app doesn't try to read from disk
      2. predict()        — so we return a fixed value without running the LSTM
    """
    with (
        patch("src.inference.predict.load_artifacts"),
        patch("src.inference.predict.predict", return_value=115.4),
    ):
        # Import app AFTER patching so lifespan uses the mock
        from api.main import app

        with TestClient(app) as c:
            yield c


# ── GET /health ───────────────────────────────────────────────────────────────


def test_health_returns_200(client):
    response = client.get("/health")
    assert response.status_code == 200


def test_health_returns_ok(client):
    response = client.get("/health")
    assert response.json() == {"status": "ok"}


# ── GET /model-info ───────────────────────────────────────────────────────────


def test_model_info_returns_200(client):
    response = client.get("/model-info")
    assert response.status_code == 200


def test_model_info_returns_json(client):
    response = client.get("/model-info")
    # Should return a dict (either metrics or the fallback config dict)
    assert isinstance(response.json(), dict)


# ── POST /predict — validation ────────────────────────────────────────────────


def test_predict_rejects_too_few_observations(client):
    """Sending fewer than 24 observations must return HTTP 422."""
    payload = {
        "observations": [
            {
                "pm25": 100,
                "temperature": 25,
                "humidity": 65,
                "wind_speed": 2,
                "pressure": 1010,
            }
        ]
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_rejects_too_many_observations(client, sample_24_observations):
    """Sending 25 observations must return HTTP 422."""
    extra = sample_24_observations + [sample_24_observations[0]]
    response = client.post("/predict", json={"observations": extra})
    assert response.status_code == 422


def test_predict_rejects_missing_field(client, sample_24_observations):
    """An observation missing 'pressure' must return HTTP 422."""
    bad = [
        {k: v for k, v in obs.items() if k != "pressure"}
        for obs in sample_24_observations
    ]
    response = client.post("/predict", json={"observations": bad})
    assert response.status_code == 422


def test_predict_rejects_wrong_type(client, sample_24_observations):
    """A non-numeric field value must return HTTP 422."""
    bad = [obs.copy() for obs in sample_24_observations]
    bad[0]["pm25"] = "not_a_number"
    response = client.post("/predict", json={"observations": bad})
    assert response.status_code == 422


# ── POST /predict — happy path ────────────────────────────────────────────────


def test_predict_returns_200(client, sample_24_observations):
    """A valid request must return HTTP 200."""
    response = client.post("/predict", json={"observations": sample_24_observations})
    assert response.status_code == 200


def test_predict_returns_predicted_pm25_key(client, sample_24_observations):
    """Response must have 'predicted_pm25' key."""
    response = client.post("/predict", json={"observations": sample_24_observations})
    assert "predicted_pm25" in response.json()


def test_predict_returns_float(client, sample_24_observations):
    """predicted_pm25 must be numeric."""
    response = client.post("/predict", json={"observations": sample_24_observations})
    value = response.json()["predicted_pm25"]
    assert isinstance(value, (int, float))


def test_predict_returns_mocked_value(client, sample_24_observations):
    """Our mock returns 115.4 — verify the API passes it through correctly."""
    with patch("api.main.predict", return_value=115.4):
        response = client.post(
            "/predict", json={"observations": sample_24_observations}
        )
    assert response.json()["predicted_pm25"] == 115.4
