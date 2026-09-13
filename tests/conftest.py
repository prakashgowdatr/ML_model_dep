"""
tests/conftest.py — Shared pytest fixtures used across all test files.

Fixtures here are automatically available to every test without importing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from sklearn.preprocessing import MinMaxScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.lstm import LSTMForecaster  # noqa: E402

# ── Constants ─────────────────────────────────────────────────────────────────
NUM_FEATURES = 5
SEQ_LEN = 24
NUM_ROWS = 200  # enough rows to build windows without real data


# ── Fake raw data ─────────────────────────────────────────────────────────────


@pytest.fixture
def fake_df():
    """A tiny DataFrame mimicking the real dataset."""
    import pandas as pd

    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "pm25": rng.uniform(10, 200, NUM_ROWS),
            "temperature": rng.uniform(5, 35, NUM_ROWS),
            "humidity": rng.uniform(20, 90, NUM_ROWS),
            "wind_speed": rng.uniform(0, 10, NUM_ROWS),
            "pressure": rng.uniform(990, 1025, NUM_ROWS),
        }
    )


@pytest.fixture
def fake_scaled_array(fake_df):
    """A scaled numpy array ready for window creation."""
    scaler = MinMaxScaler()
    return scaler.fit_transform(fake_df.values).astype(np.float32)


@pytest.fixture
def fake_scaler(fake_df):
    """A fitted MinMaxScaler on fake data."""
    scaler = MinMaxScaler()
    scaler.fit(fake_df.values)
    return scaler


@pytest.fixture
def fake_model():
    """Untrained model instance (random weights)."""
    return LSTMForecaster(num_features=NUM_FEATURES, hidden_size=32)


@pytest.fixture
def sample_batch():
    """A single fake batch tensor: (batch=4, seq=24, features=5)."""
    return torch.randn(4, SEQ_LEN, NUM_FEATURES)


@pytest.fixture
def sample_24_observations():
    """24 observation dicts suitable for the /predict endpoint."""
    rng = np.random.default_rng(0)
    return [
        {
            "pm25": float(rng.uniform(10, 200)),
            "temperature": float(rng.uniform(5, 35)),
            "humidity": float(rng.uniform(20, 90)),
            "wind_speed": float(rng.uniform(0, 10)),
            "pressure": float(rng.uniform(990, 1025)),
        }
        for _ in range(SEQ_LEN)
    ]
