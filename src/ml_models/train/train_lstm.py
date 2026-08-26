"""
train/train_lstm.py — Dev 2 (Omkar)
Train the LSTM Autoencoder on benign rows from CICIDS2018 (or UNSW-NB15).

Usage:
    python train/train_lstm.py --csv /path/to/cicids2018_benign.csv --epochs 30

Saves weights to: ml_models/models/lstm_ae_weights.pt
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Allow importing from parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))
from lstm_autoencoder import (
    LSTMAutoencoder,
    FEATURE_ORDER,
    WINDOW_SIZE,
    WEIGHTS_PATH,
    INPUT_DIM,
)

# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------
BATCH_SIZE  = 64
LR          = 1e-3
WEIGHT_DECAY = 1e-5
PATIENCE    = 5        # early-stopping patience (epochs)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def load_benign_csv(csv_path: str) -> np.ndarray:
    """
    Load a CICIDS2018 or UNSW-NB15 CSV, keep only BENIGN rows,
    extract the 11 features in FEATURE_ORDER, and return as float32 array.
    """
    df = pd.read_csv(csv_path, low_memory=False)
    df.columns = df.columns.str.strip()

    # CICIDS2018 label column variants
    for label_col in ("Label", "label", "Class", "class"):
        if label_col in df.columns:
            df = df[df[label_col].str.upper() == "BENIGN"].copy()
            break

    # Keep only features we need; fill missing with 0
    available = [f for f in FEATURE_ORDER if f in df.columns]
    missing   = [f for f in FEATURE_ORDER if f not in df.columns]
    if missing:
        print(f"[WARN] Features not found in CSV (will be zero-filled): {missing}")
        for m in missing:
            df[m] = 0.0

    X = df[FEATURE_ORDER].fillna(0).values.astype(np.float32)
    print(f"[DATA] Loaded {len(X)} benign rows from {csv_path}")
    return X


def normalise(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Min-max normalise each feature column to [0, 1]."""
    col_min = X.min(axis=0)
    col_max = X.max(axis=0)
    denom   = np.where((col_max - col_min) == 0, 1.0, col_max - col_min)
    return (X - col_min) / denom, col_min, col_max


def make_windows(X: np.ndarray, window_size: int) -> np.ndarray:
    """Slide a window over rows to create (N, window_size, features) tensor."""
    n = len(X)
    if n < window_size:
        raise ValueError(f"Not enough rows ({n}) for window_size={window_size}")
    seqs = np.stack([X[i: i + window_size] for i in range(n - window_size)], axis=0)
    return seqs   # (N-window_size, window_size, INPUT_DIM)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train(csv_path: str, epochs: int = 30) -> None:
    print(f"[TRAIN] Loading data from {csv_path} …")
    X_raw = load_benign_csv(csv_path)
    X, _, _ = normalise(X_raw)

    seqs = make_windows(X, WINDOW_SIZE)
    print(f"[TRAIN] {len(seqs)} windows of size {WINDOW_SIZE} — feature dim {INPUT_DIM}")

    # 90/10 train-val split
    split   = int(0.9 * len(seqs))
    X_train = torch.tensor(seqs[:split])
    X_val   = torch.tensor(seqs[split:])

    train_loader = DataLoader(TensorDataset(X_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val),   batch_size=BATCH_SIZE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = LSTMAutoencoder().to(device)
    opt    = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    crit   = nn.MSELoss()

    best_val_loss = float("inf")
    patience_ctr  = 0

    for epoch in range(1, epochs + 1):
        # ---- Train ----
        model.train()
        train_loss = 0.0
        for (batch,) in train_loader:
            batch = batch.to(device)
            opt.zero_grad()
            recon = model(batch)
            loss  = crit(recon, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_loss += loss.item() * len(batch)
        train_loss /= len(X_train)

        # ---- Validate ----
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (batch,) in val_loader:
                batch = batch.to(device)
                recon = model(batch)
                val_loss += crit(recon, batch).item() * len(batch)
        val_loss /= len(X_val)

        print(f"  Epoch {epoch:3d}/{epochs} | train={train_loss:.6f} | val={val_loss:.6f}")

        # ---- Early stopping ----
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_ctr  = 0
            WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), WEIGHTS_PATH)
            print(f"  [CKPT] Saved best weights → {WEIGHTS_PATH}")
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"  [STOP] Early stopping at epoch {epoch}.")
                break

    print(f"\n[DONE] Best val loss: {best_val_loss:.6f} | weights at {WEIGHTS_PATH}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LSTM Autoencoder on benign traffic")
    parser.add_argument("--csv",    required=True,  help="Path to CICIDS2018 / UNSW-NB15 CSV")
    parser.add_argument("--epochs", type=int, default=30, help="Max training epochs (default: 30)")
    args = parser.parse_args()
    train(args.csv, args.epochs)
