"""
src/models/lstm.py — Intentionally simple LSTM for PM2.5 forecasting.

Architecture:
    Input  (batch_size, 24, num_features)
      ↓
    LSTM   (hidden_size=64, num_layers=1)
      ↓  takes only the last hidden state
    Linear (64 → 1)
      ↓
    Scalar PM2.5 prediction (scaled 0-1)

That's it — no attention, no stacking, no fancy layers.
The simplicity is intentional so focus stays on the pipeline.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LSTMForecaster(nn.Module):
    """Single-layer LSTM → Linear head for one-step-ahead regression."""

    def __init__(
        self,
        num_features: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.num_features = num_features

        # LSTM layer
        # batch_first=True means input shape is (batch, seq, features)
        # instead of PyTorch's default (seq, batch, features)
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

        # Output head: map last hidden state → single prediction
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, num_features)

        Returns:
            out: (batch_size,) — one PM2.5 prediction per sample
        """
        # lstm_out: (batch, seq, hidden)
        # h_n:      (num_layers, batch, hidden)  ← last hidden state
        lstm_out, (h_n, _) = self.lstm(x)

        # Take the final hidden state of the last layer
        last_hidden = h_n[-1]  # (batch, hidden)

        # Linear projection → scalar
        out = self.head(last_hidden)  # (batch, 1)
        return out.squeeze(-1)        # (batch,)

    def count_parameters(self) -> int:
        """Helper: total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
