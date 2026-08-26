#!/usr/bin/env python3
"""K-step closed-loop rollout: no retraining, no architecture change."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.world_model.model import LSTMWorldModel
from src.world_model.prepare_dataset import generate_demo_flows
from src.world_model.windows import WindowConfig, build_state_windows
from src.world_model.rollout import (
    collect_kstep_windows,
    persist_rollout,
    rollout_closed_loop,
    summarize_kstep,
)

results = []


def check(name: str, condition: bool, detail: str = "") -> None:
    results.append((name, condition))
    icon = "PASS" if condition else "FAIL"
    extra = f"  ({detail})" if detail else ""
    print(f"  {icon}  {name}{extra}")


class CopyLast(nn.Module):
    """Toy world model: S_{t+1} = S_t. Closed loop must match persistence."""

    def forward(self, x: torch.Tensor):
        last = x[:, -1, :]
        return last, torch.zeros(x.size(0), device=x.device)


def _rebase(df: pd.DataFrame, t0: datetime) -> pd.DataFrame:
    out = df.copy()
    ts = pd.to_datetime(out["Timestamp"], dayfirst=True, utc=True)
    out["Timestamp"] = (t0 + (ts - ts.min())).dt.strftime("%d/%m/%Y %H:%M:%S")
    return out


print("\n-- Persist rollout ----------------------------------------------------")
last = np.arange(6, dtype=np.float32).reshape(2, 3)
p = persist_rollout(last, k=4)
check("persist shape", p.shape == (2, 4, 3))
check("persist copies S_t", np.allclose(p[:, 0, :], last) and np.allclose(p[:, 3, :], last))


print("\n-- Collect K-step windows (single day) --------------------------------")
df = generate_demo_flows(n_windows=30, flows_per_window=3)
bundle = build_state_windows(df, WindowConfig(window_seconds=5, horizon_k=6))
batch = collect_kstep_windows(
    bundle["states"],
    bundle["next_states"],
    bundle["timestamps"],
    bundle["day_id"],
    seq_len=8,
    k=6,
    max_gap_seconds=15.0,
)
check("history is 8 x 32", batch["x"].shape[1:] == (8, 32))
check("future is K=6 x 32", batch["y_true"].shape[1:] == (6, 32))
check("kept some k-step windows", batch["n_kept"] >= 1)
check("aligned counts", batch["x"].shape[0] == batch["y_true"].shape[0] == batch["n_kept"])
# S_{t+1} from pair i should match history's next after 8-step when rows are consecutive
if batch["n_kept"]:
    # last history state vs first future: future[0] is next_states[i], history[-1] is states[i]
    # they should differ (next vs current) on a changing demo
    check("y_true k=1 is 32-d", batch["y_true"].shape[2] == 32)
    check("same day throughout", set(map(str, batch["day_id"])) == {"2018-02-28"})


print("\n-- Closed loop CopyLast == persist ------------------------------------")
copy = CopyLast()
closed, logits = rollout_closed_loop(copy, torch.from_numpy(batch["x"][:5]), k=6)
persist = persist_rollout(batch["x"][:5, -1, :], k=6)
check("closed-loop shape (5, 6, 32)", tuple(closed.shape) == (5, 6, 32))
check("logits shape (5, 6)", tuple(logits.shape) == (5, 6))
check("CopyLast matches persist", np.allclose(closed, persist, atol=1e-5))


print("\n-- Day boundary cannot be rolled across -------------------------------")
df14 = _rebase(generate_demo_flows(n_windows=16, flows_per_window=3), datetime(2018, 2, 14, 10, 0, tzinfo=timezone.utc))
df15 = _rebase(generate_demo_flows(n_windows=16, flows_per_window=3), datetime(2018, 2, 15, 10, 0, tzinfo=timezone.utc))
b14 = build_state_windows(df14, WindowConfig(), source_file="a.csv")
b15 = build_state_windows(df15, WindowConfig(), source_file="b.csv")
from src.world_model.dataset import concat_window_bundles
comb = concat_window_bundles([b14, b15])
crossed = collect_kstep_windows(
    comb["states"], comb["next_states"], comb["timestamps"], comb["day_id"],
    seq_len=8, k=6, max_gap_seconds=15.0,
)
check("no mixed-day k-step windows", all(d in {"2018-02-14", "2018-02-15"} for d in map(str, crossed["day_id"])))
check("rejected some day-boundary futures", crossed["n_rejected_day"] >= 1)


print("\n-- Large timestamp gap cannot be rolled across ------------------------")
df_gap = generate_demo_flows(n_windows=24, flows_per_window=3)
b_gap = build_state_windows(df_gap, WindowConfig(), source_file="gap.csv")
gapped = dict(b_gap)
ts = gapped["timestamps"].copy()
ts[8:] = ts[8:] + 60.0
gapped["timestamps"] = ts
gap_batch = collect_kstep_windows(
    gapped["states"], gapped["next_states"], gapped["timestamps"], gapped["day_id"],
    seq_len=8, k=6, max_gap_seconds=15.0,
)
check("gap rejects k-step windows", gap_batch["n_rejected_gap"] >= 1)
check("still keeps within-block windows", gap_batch["n_kept"] >= 1)


print("\n-- Real LSTM architecture unchanged -----------------------------------")
model = LSTMWorldModel(input_dim=32, hidden_dim=64, num_layers=2, dropout=0.2)
model.eval()
with torch.no_grad():
    yhat, logit = rollout_closed_loop(model, torch.from_numpy(batch["x"][:2]), k=6)
check("LSTM k-step (2, 6, 32)", tuple(yhat.shape) == (2, 6, 32))
check("2x64 LSTM", model.lstm.num_layers == 2 and model.lstm.hidden_size == 64)


print("\n-- Per-k summary ------------------------------------------------------")
true = batch["y_true"][:5]
pred = persist_rollout(batch["x"][:5, -1, :], k=6)
summary = summarize_kstep(true, pred)
check("has per-k metrics", len(summary["by_k"]) == 6)
check("k=1 keys", "mse" in summary["by_k"][0] and "mae" in summary["by_k"][0])
check("overall mse present", "mse" in summary["overall"])


print("\n-- Summary ------------------------------------------------------------")
passed = sum(1 for _, ok in results if ok)
failed = sum(1 for _, ok in results if not ok)
print(f"\n  Total: {len(results)}   Passed: {passed}   Failed: {failed}\n")
if failed:
    sys.exit(1)
print("  K-step rollout helpers OK.\n")
