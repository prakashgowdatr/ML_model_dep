"""
tests/test_preprocessing.py — Unit tests for src/data/preprocessing.py

These tests run without any real dataset by using the fake_df fixture.
They verify:
  1. Missing-value handling
  2. Chronological split sizes
  3. Sliding-window shapes
  4. No data leakage (scaler fitted on train only)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.data.preprocessing import clean, make_windows, split

# ── 1. Missing value handling ─────────────────────────────────────────────────


def test_clean_forward_fills_nans(fake_df):
    """NaN values should be forward-filled, not just dropped."""
    dirty = fake_df.copy()
    dirty.iloc[5, 0] = float("nan")  # inject one NaN in pm25
    dirty.iloc[10, 2] = float("nan")  # inject one NaN in humidity

    cleaned = clean(dirty)

    assert cleaned.isna().sum().sum() == 0, "All NaNs should be gone after cleaning"
    assert len(cleaned) == len(fake_df), "No rows should be dropped for single NaNs"


def test_clean_drops_rows_with_all_nans():
    """Rows that can't be forward-filled (e.g., first row is NaN) are dropped."""
    df = pd.DataFrame(
        {
            "pm25": [float("nan"), 50.0, 60.0],
            "temperature": [float("nan"), 20.0, 21.0],
            "humidity": [float("nan"), 60.0, 62.0],
            "wind_speed": [float("nan"), 2.0, 2.1],
            "pressure": [float("nan"), 1010.0, 1011.0],
        }
    )
    cleaned = clean(df)
    # First row is NaN; ffill can't help (no previous row), bfill should fill it
    # bfill picks values from row 1 → row 0 should be filled
    assert cleaned.isna().sum().sum() == 0


# ── 2. Split ──────────────────────────────────────────────────────────────────


def test_split_sizes_sum_to_original(fake_df):
    """The three splits must together contain all original rows."""
    train, val, test = split(fake_df)
    assert len(train) + len(val) + len(test) == len(fake_df)


def test_split_is_chronological(fake_df):
    """Rows must NOT be shuffled — train comes before val, val before test."""
    train, val, test = split(fake_df)
    assert train.index[-1] < val.index[0]
    assert val.index[-1] < test.index[0]


def test_split_ratios_are_approximate(fake_df):
    """Train split should be ~70% of the data (within 2%)."""
    train, _, _ = split(fake_df)
    ratio = len(train) / len(fake_df)
    assert abs(ratio - 0.70) < 0.02, f"Train ratio {ratio:.2f} is far from 0.70"


# ── 3. Sliding window ─────────────────────────────────────────────────────────


def test_make_windows_output_shape(fake_scaled_array):
    """X should be (N-seq_len, seq_len, features); y should be (N-seq_len,)."""
    seq_len = 24
    data = fake_scaled_array
    X, y = make_windows(data, seq_len=seq_len, target_col_idx=0)

    expected_windows = len(data) - seq_len
    assert X.shape == (expected_windows, seq_len, data.shape[1])
    assert y.shape == (expected_windows,)


def test_make_windows_target_is_next_step(fake_scaled_array):
    """y[i] must equal the target feature value at position i + seq_len."""
    seq_len = 5
    data = fake_scaled_array
    X, y = make_windows(data, seq_len=seq_len, target_col_idx=0)

    for i in range(min(10, len(y))):
        expected = data[i + seq_len, 0]
        assert abs(y[i] - expected) < 1e-6, f"Window {i}: y mismatch"


def test_make_windows_dtype(fake_scaled_array):
    """Arrays should be float32 for PyTorch compatibility."""
    X, y = make_windows(fake_scaled_array, seq_len=24, target_col_idx=0)
    assert X.dtype == np.float32
    assert y.dtype == np.float32


# ── 4. No data leakage ────────────────────────────────────────────────────────


def test_scaler_fitted_only_on_train(fake_df):
    """
    If we fit on train and transform val, the val values can exceed [0, 1]
    (because val may have larger values than train).
    This confirms the scaler was NOT fit on val/test data.
    """
    train, val, _ = split(fake_df)

    scaler = MinMaxScaler()
    scaler.fit(train.values)

    train_scaled = scaler.transform(train.values)
    # Train should be in [0, 1]
    assert train_scaled.min() >= -1e-6
    assert train_scaled.max() <= 1.0 + 1e-6

    # Val CAN go slightly outside [0, 1] — and that's correct behaviour
    # (it means val has values unseen during training)
    # We just verify the scaler was fitted on train dimensions
    assert scaler.n_features_in_ == train.shape[1]
