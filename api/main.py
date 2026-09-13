"""
api/main.py — FastAPI application for PM2.5 forecasting.

Endpoints:
  GET  /health       → liveness check
  GET  /model-info   → metadata about the loaded model
  POST /predict      → main prediction endpoint

Run locally:
    uvicorn api.main:app --reload
"""

from __future__ import annotations

import json
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator

# ── Make project root importable ─────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CFG                   # noqa: E402
from src.inference.predict import load_artifacts, predict  # noqa: E402


# ── Pydantic schemas ─────────────────────────────────────────────────────────

class Observation(BaseModel):
    """One hour of sensor readings."""
    pm25:        float
    temperature: float
    humidity:    float
    wind_speed:  float
    pressure:    float


class PredictRequest(BaseModel):
    """
    Request body for POST /predict.
    Must contain exactly 24 hourly observations.
    """
    observations: List[Observation]

    @field_validator("observations")
    @classmethod
    def must_be_24(cls, v: List[Observation]) -> List[Observation]:
        seq_len = CFG.preprocessing.sequence_length  # 24
        if len(v) != seq_len:
            raise ValueError(
                f"Exactly {seq_len} observations are required, got {len(v)}."
            )
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "observations": [
                    {
                        "pm25": round(45.0 + (i % 10) * 3.0, 1),
                        "temperature": round(16.0 + (i % 8) * 1.2, 1),
                        "humidity": round(50.0 + (i % 6) * 2.5, 1),
                        "wind_speed": round(1.5 + (i % 5) * 0.4, 1),
                        "pressure": round(1012.0 - (i % 4) * 0.8, 1),
                    }
                    for i in range(24)
                ]
            }
        }
    }


class PredictResponse(BaseModel):
    """Response body for POST /predict."""
    predicted_pm25: float


# ── App lifecycle ─────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan event — runs BEFORE the server accepts requests.
    We load the model here so every request is fast (no repeated disk I/O).
    """
    load_artifacts()
    yield
    # (cleanup code would go here if needed)


# ── Create app ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="PM2.5 Forecaster",
    description="LSTM-based next-hour PM2.5 prediction API",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", tags=["System"])
def root():
    """Welcome endpoint providing quick links to documentation and status."""
    return {
        "title": "PM2.5 Forecaster API",
        "status": "running",
        "docs_url": "/docs",
        "health_url": "/health",
        "model_info_url": "/model-info",
    }


@app.get("/health", tags=["System"])
def health():
    """Liveness check — CI and monitoring ping this endpoint."""
    return {"status": "ok"}


@app.get("/model-info", tags=["System"])
def model_info():
    """Return metadata about the loaded model."""
    metrics_path = PROJECT_ROOT / CFG.artifacts.metrics_path
    if not metrics_path.exists():
        return {
            "message": "Model not yet trained or metrics file missing.",
            "model_config": {
                "hidden_size":      CFG.model.hidden_size,
                "num_layers":       CFG.model.num_layers,
                "sequence_length":  CFG.preprocessing.sequence_length,
                "features":         CFG.preprocessing.features,
            },
        }
    with open(metrics_path) as f:
        return json.load(f)


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict_endpoint(request: PredictRequest):
    """
    Predict next-hour PM2.5 from the past 24 hours of observations.

    Example request body:
    {
      "observations": [
        {"pm25": 120, "temperature": 27.5, "humidity": 70,
         "wind_speed": 2.1, "pressure": 1008},
        ... (24 entries total)
      ]
    }
    """
    try:
        obs_dicts = [obs.model_dump() for obs in request.observations]
        result    = predict(obs_dicts)
        return PredictResponse(predicted_pm25=result)
    except RuntimeError as e:
        # Model not loaded (e.g., artifacts missing)
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
