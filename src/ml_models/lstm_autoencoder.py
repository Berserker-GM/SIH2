"""
lstm_autoencoder.py — Dev 2 (Omkar)
LSTM Autoencoder for sequence-based anomaly detection.

Consumes sliding windows of ML_FEATURE_VECTOR dicts (per host_key),
computes reconstruction error, and returns a normalized anomaly score.

Feature order MUST match Dev 1 feature_extractor.py FEATURE_ORDER exactly.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Feature contract — sync this list with Dev 1 before Day 2
# ---------------------------------------------------------------------------
FEATURE_ORDER = [
    "login_fail_count",
    "login_success_count",
    "unique_dest_ips",
    "bytes_sent_total",
    "file_ops_count",
    "cpu_pct_avg",
    "payload_flag_count",
    "inter_arrival_ms_avg",
    "entropy_dest_ports",
    "src_port",       # must arrive pre-normalised from Dev 1
    "dest_port",      # must arrive pre-normalised from Dev 1
]

INPUT_DIM    = len(FEATURE_ORDER)   # 11
HIDDEN_DIM   = 64
LATENT_DIM   = 16
NUM_LAYERS   = 2
WINDOW_SIZE  = 20                   # events per sequence
ANOMALY_THRESHOLD = 0.6             # reconstruction error above this → anomaly

WEIGHTS_PATH = Path(__file__).parent / "models" / "lstm_ae_weights.pt"


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class LSTMEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, num_layers: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        _, (h_n, _) = self.lstm(x)
        # h_n: (num_layers, batch, hidden_dim) → take top layer
        latent = self.fc(h_n[-1])          # (batch, latent_dim)
        return latent


class LSTMDecoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int,
                 num_layers: int, seq_len: int):
        super().__init__()
        self.seq_len = seq_len
        self.fc = nn.Linear(latent_dim, hidden_dim)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0,
        )
        self.output_fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, latent):
        # Expand latent to (batch, seq_len, hidden_dim)
        h = self.fc(latent).unsqueeze(1).repeat(1, self.seq_len, 1)
        out, _ = self.lstm(h)
        reconstructed = self.output_fc(out)   # (batch, seq_len, output_dim)
        return reconstructed


class LSTMAutoencoder(nn.Module):
    def __init__(
        self,
        input_dim: int  = INPUT_DIM,
        hidden_dim: int = HIDDEN_DIM,
        latent_dim: int = LATENT_DIM,
        num_layers: int = NUM_LAYERS,
        seq_len: int    = WINDOW_SIZE,
    ):
        super().__init__()
        self.encoder = LSTMEncoder(input_dim, hidden_dim, latent_dim, num_layers)
        self.decoder = LSTMDecoder(latent_dim, hidden_dim, input_dim, num_layers, seq_len)

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample mean-squared reconstruction error. Shape: (batch,)"""
        recon = self.forward(x)
        return ((x - recon) ** 2).mean(dim=(1, 2))


# ---------------------------------------------------------------------------
# Singleton loader
# ---------------------------------------------------------------------------
_model: LSTMAutoencoder | None = None

# Running stats for online min-max normalisation of reconstruction error
_re_min: float = 0.0
_re_max: float = 1.0
_re_ema_alpha: float = 0.01   # smoothing factor for online update


def load_model(weights_path: Path = WEIGHTS_PATH) -> LSTMAutoencoder:
    """Load (or create) the autoencoder. Call once at service startup."""
    global _model
    if _model is not None:
        return _model

    _model = LSTMAutoencoder()
    if weights_path.exists():
        _model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        print(f"[LSTM-AE] Loaded weights from {weights_path}")
    else:
        print(f"[LSTM-AE] WARNING — no weights found at {weights_path}. "
              "Run train/train_lstm.py before deploying.")

    _model.eval()
    return _model


def _update_normalisation_stats(raw_error: float) -> None:
    """Online EMA update for min/max normalisation bounds."""
    global _re_min, _re_max
    _re_min = (1 - _re_ema_alpha) * _re_min + _re_ema_alpha * min(raw_error, _re_min)
    _re_max = (1 - _re_ema_alpha) * _re_max + _re_ema_alpha * max(raw_error, _re_max)


def feature_vectors_to_tensor(feature_vectors: list[dict]) -> torch.Tensor:
    """
    Convert a list of ML_FEATURE_VECTOR dicts to a (1, seq_len, INPUT_DIM) tensor.
    Missing keys default to 0.0.
    """
    rows = []
    for fv in feature_vectors:
        row = [float(fv.get(k, 0.0)) for k in FEATURE_ORDER]
        rows.append(row)
    arr = np.array(rows, dtype=np.float32)               # (seq_len, INPUT_DIM)
    return torch.tensor(arr).unsqueeze(0)                # (1, seq_len, INPUT_DIM)


def score_window_lstm(feature_vectors: list[dict]) -> float:
    """
    Score a window of feature vectors with the LSTM Autoencoder.

    Returns:
        lstm_score (float): reconstruction error normalised to [0.0, 1.0].
                            Values > ANOMALY_THRESHOLD indicate anomaly.
    """
    model = load_model()
    x = feature_vectors_to_tensor(feature_vectors)

    with torch.no_grad():
        raw_error = model.reconstruction_error(x).item()

    _update_normalisation_stats(raw_error)

    # Clip-normalise to [0, 1]
    denom = max(_re_max - _re_min, 1e-8)
    score = float(np.clip((raw_error - _re_min) / denom, 0.0, 1.0))
    return score
