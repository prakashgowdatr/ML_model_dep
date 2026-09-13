"""
src/evaluation/evaluate.py — Load saved model and score on the test set.

Metrics reported:
  • MAE  — average absolute error (in scaled units, then un-scaled)
  • RMSE — root mean squared error
  • R²   — coefficient of determination (1.0 = perfect)

Run:
    python src/evaluation/evaluate.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CFG  # noqa: E402
from src.data.preprocessing import load_scaler, run_pipeline  # noqa: E402
from src.models.lstm import LSTMForecaster  # noqa: E402


def load_model(num_features: int, device: torch.device) -> LSTMForecaster:
    """Reconstruct model architecture and load saved weights."""
    model = LSTMForecaster(
        num_features=num_features,
        hidden_size=CFG.model.hidden_size,
        num_layers=CFG.model.num_layers,
        dropout=CFG.model.dropout,
    ).to(device)

    model_path = PROJECT_ROOT / CFG.artifacts.model_path
    # map_location ensures it works even if trained on GPU but evaluated on CPU
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


@torch.no_grad()
def get_predictions(
    model: LSTMForecaster,
    X: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Run inference in batches.  Returns scaled predictions."""
    X_tensor = torch.from_numpy(X).to(device)
    preds = model(X_tensor).cpu().numpy()
    return preds


def inverse_transform_pm25(
    scaled_values: np.ndarray,
    scaler,
    num_features: int,
    target_idx: int,
) -> np.ndarray:
    """
    Un-scale PM2.5 predictions back to original µg/m³ units.

    The scaler was fit on all features together, so we need to build
    a dummy array of the right shape and extract the target column.
    """
    dummy = np.zeros((len(scaled_values), num_features), dtype=np.float32)
    dummy[:, target_idx] = scaled_values
    unscaled = scaler.inverse_transform(dummy)
    return unscaled[:, target_idx]


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load data ────────────────────────────────────────────────────────────
    print("📂  Running preprocessing pipeline …")
    data = run_pipeline()
    scaler = load_scaler()
    X_test = data["X_test"]
    y_test = data["y_test"]

    num_features = X_test.shape[2]
    features = CFG.preprocessing.features
    target_idx = features.index(CFG.preprocessing.target)

    # ── Load model ───────────────────────────────────────────────────────────
    print("🧠  Loading model …")
    model = load_model(num_features, device)

    # ── Inference ────────────────────────────────────────────────────────────
    y_pred_scaled = get_predictions(model, X_test, device)

    # ── Un-scale ─────────────────────────────────────────────────────────────
    y_pred = inverse_transform_pm25(y_pred_scaled, scaler, num_features, target_idx)
    y_true = inverse_transform_pm25(y_test, scaler, num_features, target_idx)

    # ── Metrics ──────────────────────────────────────────────────────────────
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    print("\n📊  Test Set Metrics")
    print(f"    MAE  : {mae:.2f} µg/m³")
    print(f"    RMSE : {rmse:.2f} µg/m³")
    print(f"    R²   : {r2:.4f}")

    # ── Save metrics to disk ─────────────────────────────────────────────────
    metrics_path = PROJECT_ROOT / CFG.artifacts.metrics_path
    existing = {}
    if metrics_path.exists():
        try:
            with open(metrics_path) as f:
                existing = json.load(f)
        except json.JSONDecodeError:
            existing = {}

    existing.update(
        {
            "test_mae": round(float(mae), 4),
            "test_rmse": round(float(rmse), 4),
            "test_r2": round(float(r2), 4),
        }
    )
    with open(metrics_path, "w") as f:
        json.dump(existing, f, indent=2)

    print(f"\n✅  Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
