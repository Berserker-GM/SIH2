#!/usr/bin/env python3
"""Tests for LSTM world-model training helpers (no full CIC run)."""

import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.world_model.dataset import fit_scaler, make_sequences, temporal_split
from src.world_model.metrics import best_f1_threshold, classification_metrics
from src.world_model.model import LSTMWorldModel
from src.world_model.prepare_dataset import generate_demo_flows
from src.world_model.windows import WindowConfig, build_state_windows

results = []


def check(name: str, condition: bool, detail: str = "") -> None:
    results.append((name, condition))
    icon = "PASS" if condition else "FAIL"
    extra = f"  ({detail})" if detail else ""
    print(f"  {icon}  {name}{extra}")


print("\n-- Temporal split -----------------------------------------------------")
tr, va, te = temporal_split(100)
check("contiguous train", np.array_equal(tr, np.arange(70)))
check("no index overlap", len(set(tr) & set(va) & set(te)) == 0 and len(set(tr) & set(va)) == 0)
check("covers all rows", len(tr) + len(va) + len(te) == 100)
check("time order preserved", tr[-1] < va[0] < te[0])


print("\n-- Scaler fit on train only ------------------------------------------")
rng = np.random.default_rng(0)
train = rng.normal(10, 2, size=(50, 32))
test = rng.normal(30, 2, size=(20, 32))
scaler = fit_scaler(train)
z_train = scaler.transform(train)
z_test = scaler.transform(test)
check("train ~ zero mean", abs(float(z_train.mean())) < 0.05)
check("test not zero-centered (no leakage)", abs(float(z_test.mean())) > 1.0)


print("\n-- Sequences and forward pass ----------------------------------------")
df = generate_demo_flows(n_windows=24, flows_per_window=4)
bundle = build_state_windows(df, WindowConfig(window_seconds=5, horizon_k=6))
seq_len = 4
x, y_next, y_atk = make_sequences(bundle["states"], bundle["next_states"], bundle["attack_within_k"], seq_len)
check("sequence shape", x.shape[1:] == (seq_len, 32))
check("next-state rows match", y_next.shape[0] == x.shape[0] and y_next.shape[1] == 32)

model = LSTMWorldModel(input_dim=32, hidden_dim=16, num_layers=1)
model.eval()
with torch.no_grad():
    nxt, logit = model(torch.from_numpy(x[:3]))
check("next_state head (3, 32)", tuple(nxt.shape) == (3, 32))
check("attack logit (3,)", tuple(logit.shape) == (3,))


print("\n-- Metrics ------------------------------------------------------------")
yt = np.array([0, 0, 1, 1, 1, 0])
yp = np.array([0, 1, 1, 1, 0, 0])
m = classification_metrics(yt, yp)
check("fpr uses tn/fp", abs(m["fpr"] - (1 / 3)) < 1e-6)
check("threshold search returns in (0,1)", 0.05 <= best_f1_threshold(yt, np.array([0.1, 0.4, 0.6, 0.9, 0.2, 0.3])) <= 0.95)


print("\n-- Summary ------------------------------------------------------------")
passed = sum(1 for _, ok in results if ok)
failed = sum(1 for _, ok in results if not ok)
print(f"\n  Total: {len(results)}   Passed: {passed}   Failed: {failed}\n")
if failed:
    sys.exit(1)
print("  World-model train helpers OK.\n")
