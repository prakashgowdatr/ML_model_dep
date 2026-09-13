"""
src/inference/predict.py — Reusable prediction function used by the API.

This module is the bridge between the raw input (24 observations from the
HTTP request) and the trained LSTM model.

Steps:
  1. Accept a list of 24 observation dicts
  2. Convert to numpy array (shape 24 x 5)
  3. Scale using the saved scaler
  4. Run LSTM forward pass
  5. Un-scale the PM2.5 output
  6. Return a float
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CFG  # noqa: E402
from src.models.lstm import LSTMForecaster  # noqa: E402

# ── Lazy-loaded singletons (loaded once when the API starts) ─────────────────
_model: LSTMForecaster | None = None
_scaler = None
_device: torch.device | None = None


def _get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_artifacts() -> None:
    """
    Load model weights and scaler from disk.
    Called once at API startup — not on every request.
    """
    global _model, _scaler, _device

    _device = _get_device()

    # Load scaler
    scaler_path = PROJECT_ROOT / CFG.artifacts.scaler_path
    with open(scaler_path, "rb") as f:
        _scaler = pickle.load(f)

    # Reconstruct model and load weights
    num_features = len(CFG.preprocessing.features)
    _model = LSTMForecaster(
        num_features=num_features,
        hidden_size=CFG.model.hidden_size,
        num_layers=CFG.model.num_layers,
        dropout=CFG.model.dropout,
    ).to(_device)

    model_path = PROJECT_ROOT / CFG.artifacts.model_path
    state_dict = torch.load(model_path, map_location=_device)
    _model.load_state_dict(state_dict)
    _model.eval()  # disable dropout, etc.

    print(f"✅  Model loaded from {model_path}")
    print(f"✅  Scaler loaded from {scaler_path}")


def predict(observations: list[dict]) -> float:
    """
    Predict next-hour PM2.5 from 24 hourly observations.

    Args:
        observations: list of 24 dicts, each with keys:
                      pm25, temperature, humidity, wind_speed, pressure

    Returns:
        Predicted PM2.5 concentration in µg/m³ (float)
    """
    if _model is None or _scaler is None:
        raise RuntimeError("Call load_artifacts() before predict().")

    features = CFG.preprocessing.features
    target = CFG.preprocessing.target
    target_idx = features.index(target)
    num_features = len(features)

    # ── Build input array (24, 5) ────────────────────────────────────────────
    arr = np.array(
        [[obs[feat] for feat in features] for obs in observations],
        dtype=np.float32,
    )  # shape: (24, 5)

    # ── Scale ────────────────────────────────────────────────────────────────
    arr_scaled = _scaler.transform(arr)  # (24, 5)

    # ── Add batch dimension → (1, 24, 5) ─────────────────────────────────────
    X = torch.from_numpy(arr_scaled[np.newaxis, ...]).to(_device)

    # ── Inference ─────────────────────────────────────────────────────────────
    with torch.no_grad():
        pred_scaled = _model(X).item()  # scalar (still in 0-1 range)

    # ── Un-scale PM2.5 ────────────────────────────────────────────────────────
    dummy = np.zeros((1, num_features), dtype=np.float32)
    dummy[0, target_idx] = pred_scaled
    pred_original = _scaler.inverse_transform(dummy)[0, target_idx]

    return float(round(pred_original, 2))
