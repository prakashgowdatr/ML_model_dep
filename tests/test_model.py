"""
tests/test_model.py — Unit tests for the LSTM model.

These tests do NOT require a trained model.
They verify architecture and forward-pass behaviour
using random weights (just model instantiation).
"""

import pytest
import torch

from src.models.lstm import LSTMForecaster

# ── 1. Instantiation ──────────────────────────────────────────────────────────


def test_model_instantiates():
    """Model should be created without errors."""
    model = LSTMForecaster(num_features=5)
    assert model is not None


def test_model_has_lstm_and_head():
    """Verify the expected submodules exist."""
    model = LSTMForecaster(num_features=5, hidden_size=32)
    assert hasattr(model, "lstm")
    assert hasattr(model, "head")


def test_model_parameter_count_is_reasonable():
    """
    For num_features=5, hidden=64:
    LSTM params ≈ 4 * (64*(5+64) + 64) ≈ 17,920
    Linear params = 64 + 1 = 65
    Total ≈ ~18k — small, as intended.
    """
    model = LSTMForecaster(num_features=5, hidden_size=64)
    total = model.count_parameters()
    # Should be well under 100k for our simple architecture
    assert total < 100_000, f"Model is too large: {total} params"
    assert total > 0


# ── 2. Forward pass shapes ────────────────────────────────────────────────────


def test_forward_output_shape_batch_of_4(fake_model, sample_batch):
    """Output shape should be (batch_size,) — one scalar per sample."""
    out = fake_model(sample_batch)
    assert out.shape == (4,), f"Expected (4,), got {out.shape}"


def test_forward_output_shape_batch_of_1(fake_model):
    """Single-sample inference should return shape (1,)."""
    x = torch.randn(1, 24, 5)
    out = fake_model(x)
    assert out.shape == (1,)


def test_forward_output_is_finite(fake_model, sample_batch):
    """Output should never be NaN or Inf with random weights."""
    out = fake_model(sample_batch)
    assert torch.all(torch.isfinite(out)), "Model output contains NaN or Inf"


# ── 3. Different configurations ───────────────────────────────────────────────


@pytest.mark.parametrize("hidden_size", [16, 32, 64, 128])
def test_forward_different_hidden_sizes(hidden_size):
    """Model should work for any hidden size."""
    model = LSTMForecaster(num_features=5, hidden_size=hidden_size)
    x = torch.randn(2, 24, 5)
    out = model(x)
    assert out.shape == (2,)


@pytest.mark.parametrize("seq_len", [12, 24, 48])
def test_forward_different_sequence_lengths(seq_len):
    """LSTM accepts any sequence length — not hardcoded."""
    model = LSTMForecaster(num_features=5, hidden_size=32)
    x = torch.randn(2, seq_len, 5)
    out = model(x)
    assert out.shape == (2,)


def test_forward_different_feature_counts():
    """Model should adapt to different numbers of input features."""
    for n_feat in [3, 5, 10]:
        model = LSTMForecaster(num_features=n_feat, hidden_size=32)
        x = torch.randn(2, 24, n_feat)
        out = model(x)
        assert out.shape == (2,)


# ── 4. Gradient flow ──────────────────────────────────────────────────────────


def test_gradients_flow_through_model(fake_model, sample_batch):
    """
    After a backward pass, all parameters should have gradients.
    This confirms the model is fully differentiable.
    """
    out = fake_model(sample_batch)
    loss = out.mean()
    loss.backward()

    for name, param in fake_model.named_parameters():
        assert param.grad is not None, f"No gradient for parameter: {name}"
        assert torch.all(torch.isfinite(param.grad)), f"Inf/NaN gradient in: {name}"
