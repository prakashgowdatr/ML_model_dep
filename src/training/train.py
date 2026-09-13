"""
src/training/train.py — Train the LSTM and save the best checkpoint.

Run:
    python src/training/train.py

What this script teaches:
  • PyTorch Dataset / DataLoader pattern
  • Training loop (forward → loss → backward → step)
  • Validation loop (no gradients, evaluation only)
  • Early stopping (stop training when val loss stops improving)
  • Model checkpointing (save only the best model weights)
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CFG                         # noqa: E402
from src.data.preprocessing import run_pipeline    # noqa: E402
from src.models.lstm import LSTMForecaster         # noqa: E402


# ── Reproducibility ───────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    """Fix random seeds so results are reproducible across runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Dataset helper ────────────────────────────────────────────────────────────

def make_loader(X: np.ndarray, y: np.ndarray, shuffle: bool) -> DataLoader:
    """Wrap numpy arrays into a PyTorch DataLoader."""
    dataset = TensorDataset(
        torch.from_numpy(X),  # (N, seq_len, features)
        torch.from_numpy(y),  # (N,)
    )
    return DataLoader(
        dataset,
        batch_size=CFG.training.batch_size,
        shuffle=shuffle,
        # pin_memory speeds up GPU transfer; harmless on CPU
        pin_memory=torch.cuda.is_available(),
    )


# ── Training loop ─────────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Run one full pass over the training data.  Returns mean loss."""
    model.train()  # activates dropout / batch norm (none here, but good habit)
    total_loss = 0.0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        # Forward pass
        predictions = model(X_batch)

        # Compute loss
        loss = criterion(predictions, y_batch)

        # Backward pass
        optimizer.zero_grad()  # clear old gradients
        loss.backward()        # compute new gradients
        optimizer.step()       # update weights

        total_loss += loss.item() * len(X_batch)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Evaluate without computing gradients.  Returns mean loss."""
    model.eval()
    total_loss = 0.0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        predictions = model(X_batch)
        loss = criterion(predictions, y_batch)
        total_loss += loss.item() * len(X_batch)

    return total_loss / len(loader.dataset)


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    set_seed(CFG.training.seed)

    # ── Device (GPU if available, else CPU) ──────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️   Using device: {device}")

    # ── Data ─────────────────────────────────────────────────────────────────
    data = run_pipeline()
    train_loader = make_loader(data["X_train"], data["y_train"], shuffle=True)
    val_loader   = make_loader(data["X_val"],   data["y_val"],   shuffle=False)

    num_features = data["X_train"].shape[2]  # number of input columns

    # ── Model ─────────────────────────────────────────────────────────────────
    model = LSTMForecaster(
        num_features=num_features,
        hidden_size=CFG.model.hidden_size,
        num_layers=CFG.model.num_layers,
        dropout=CFG.model.dropout,
    ).to(device)
    print(f"🧠  Parameters: {model.count_parameters():,}")

    # ── Loss & optimizer ──────────────────────────────────────────────────────
    # HuberLoss is MSE for small errors, MAE for large ones (robust to outliers)
    criterion = nn.HuberLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG.training.learning_rate)

    # ── Training loop with early stopping ────────────────────────────────────
    best_val_loss = float("inf")
    patience_counter = 0
    model_path = PROJECT_ROOT / CFG.artifacts.model_path
    model_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'Epoch':>5}  {'Train Loss':>12}  {'Val Loss':>10}  {'Status':>10}")
    print("-" * 48)

    for epoch in range(1, CFG.training.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss   = evaluate(model, val_loader, criterion, device)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save only the model weights, not the whole model object
            torch.save(model.state_dict(), model_path)
            status = "✅ saved"
        else:
            patience_counter += 1
            status = f"⏳ {patience_counter}/{CFG.training.early_stopping_patience}"

        print(f"{epoch:>5}  {train_loss:>12.6f}  {val_loss:>10.6f}  {status:>10}")

        if patience_counter >= CFG.training.early_stopping_patience:
            print(f"\n⏹️   Early stopping at epoch {epoch}")
            break

    print(f"\n✅  Best model saved to {model_path}")
    print(f"    Best val loss: {best_val_loss:.6f}")

    # Save lightweight metadata for the API /model-info endpoint
    info = {
        "best_val_loss": best_val_loss,
        "num_features":  num_features,
        "hidden_size":   CFG.model.hidden_size,
        "num_layers":    CFG.model.num_layers,
        "sequence_length": CFG.preprocessing.sequence_length,
        "features":      CFG.preprocessing.features,
    }
    metrics_path = PROJECT_ROOT / CFG.artifacts.metrics_path
    with open(metrics_path, "w") as f:
        json.dump(info, f, indent=2)
    print(f"    Model info saved to {metrics_path}")

    print("\n🎉  Training complete!  Next steps:")
    print("    python src/evaluation/evaluate.py   ← check MAE / RMSE / R²")
    print("    uvicorn api.main:app --reload        ← start the API")


if __name__ == "__main__":
    main()
