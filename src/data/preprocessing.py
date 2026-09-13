"""
src/data/preprocessing.py — Build train/val/test datasets from raw CSV.

What this module does, step by step:
  1. Load the CSV
  2. Forward-fill missing values (avoids data leakage — no future info)
  3. Sort chronologically
  4. Split 70 / 15 / 15  (no shuffling — time order must be preserved)
  5. Fit a MinMaxScaler on the TRAIN split only
  6. Build 24-hour sliding windows → shape (N, 24, num_features)
  7. Save the fitted scaler so the API can use it later

Run standalone:
    python src/data/preprocessing.py
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CFG  # noqa: E402

# ── 1. Load ──────────────────────────────────────────────────────────────────


def load_raw(path: str | Path | None = None) -> pd.DataFrame:
    """Load the raw CSV.  Returns a DataFrame with only the feature columns."""
    path = path or PROJECT_ROOT / CFG.data.raw_path
    df = pd.read_csv(path)

    # Keep only features we care about
    cols = CFG.preprocessing.features
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"CSV is missing columns: {missing}\n"
            "Run: python scripts/download_data.py"
        )

    return df[cols].copy()


# ── 2. Clean ─────────────────────────────────────────────────────────────────


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill then back-fill NaNs; drop any remaining rows."""
    df = df.ffill().bfill()
    n_dropped = df.isna().any(axis=1).sum()
    if n_dropped > 0:
        print(f"⚠️   Dropped {n_dropped} rows with unfillable NaNs")
        df = df.dropna()
    return df.reset_index(drop=True)


# ── 3. Split ─────────────────────────────────────────────────────────────────


def split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological 70 / 15 / 15 split.  NO shuffling."""
    n = len(df)
    t1 = int(n * CFG.preprocessing.train_ratio)
    t2 = t1 + int(n * CFG.preprocessing.val_ratio)
    return df.iloc[:t1], df.iloc[t1:t2], df.iloc[t2:]


# ── 4. Scale ─────────────────────────────────────────────────────────────────


def fit_scaler(train_df: pd.DataFrame) -> MinMaxScaler:
    """Fit MinMaxScaler on training data only.  Saves to disk."""
    scaler = MinMaxScaler()
    scaler.fit(train_df.values)

    scaler_path = PROJECT_ROOT / CFG.artifacts.scaler_path
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    print(f"✅  Scaler saved to {scaler_path}")
    return scaler


def load_scaler() -> MinMaxScaler:
    """Load a previously saved scaler."""
    scaler_path = PROJECT_ROOT / CFG.artifacts.scaler_path
    with open(scaler_path, "rb") as f:
        return pickle.load(f)


def apply_scaler(df: pd.DataFrame, scaler: MinMaxScaler) -> np.ndarray:
    """Transform a DataFrame using a fitted scaler."""
    return scaler.transform(df.values)


# ── 5. Sliding window ─────────────────────────────────────────────────────────


def make_windows(
    data: np.ndarray,
    seq_len: int | None = None,
    target_col_idx: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build input/target pairs using a sliding window.

    Args:
        data          : scaled array of shape (N, num_features)
        seq_len       : number of past hours to use as input (default from config)
        target_col_idx: index of the PM2.5 column in `data`

    Returns:
        X : shape (num_windows, seq_len, num_features)
        y : shape (num_windows,)  — next-step PM2.5 value (scaled)
    """
    seq_len = seq_len or CFG.preprocessing.sequence_length

    X_list, y_list = [], []
    for i in range(len(data) - seq_len):
        X_list.append(data[i : i + seq_len])  # 24 hours of all features
        y_list.append(data[i + seq_len, target_col_idx])  # next hour's PM2.5

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)


# ── 6. Full pipeline ──────────────────────────────────────────────────────────


def run_pipeline(
    raw_path: str | Path | None = None,
) -> dict:
    """
    End-to-end preprocessing.  Returns a dict with split numpy arrays.

    Keys: X_train, y_train, X_val, y_val, X_test, y_test, scaler
    """
    print("📂  Loading data …")
    df = load_raw(raw_path)

    print("🧹  Cleaning …")
    df = clean(df)

    print("✂️   Splitting …")
    train_df, val_df, test_df = split(df)
    print(f"    Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    print("📏  Scaling (fit on train only) …")
    scaler = fit_scaler(train_df)

    train_scaled = apply_scaler(train_df, scaler)
    val_scaled = apply_scaler(val_df, scaler)
    test_scaled = apply_scaler(test_df, scaler)

    # Target column index — PM2.5 is the first feature in our list
    features = CFG.preprocessing.features
    target_idx = features.index(CFG.preprocessing.target)

    print("🪟  Building sliding windows …")
    X_train, y_train = make_windows(train_scaled, target_col_idx=target_idx)
    X_val, y_val = make_windows(val_scaled, target_col_idx=target_idx)
    X_test, y_test = make_windows(test_scaled, target_col_idx=target_idx)

    print(
        f"    X_train: {X_train.shape},  X_val: {X_val.shape},  X_test: {X_test.shape}"
    )

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "scaler": scaler,
    }


if __name__ == "__main__":
    data = run_pipeline()
    print("\n🎉  Preprocessing complete!  Next step:")
    print("    python src/training/train.py")
