"""LSTM world model: P(S_{t+1} | S_{t-L+1:t}) and P(attack_within_k | history)."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.world_model.cic_schema import INPUT_DIM


class LSTMWorldModel(nn.Module):
    """
    Sequence model over network-state windows.

    This is NOT an autoencoder. The two heads are:
      - next_state: predicted S_{t+1}  (dynamics)
      - attack_logit: P(attack in next K windows)
    """

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.next_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
        )
        self.attack_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (batch, seq_len, input_dim)
        out, _ = self.lstm(x)
        h = out[:, -1, :]
        next_state = self.next_head(h)
        attack_logit = self.attack_head(h).squeeze(-1)
        return next_state, attack_logit
