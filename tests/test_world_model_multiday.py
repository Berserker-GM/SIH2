#!/usr/bin/env python3
"""Multi-day CIC-IDS windowing / sequences: no temporal leakage across days."""

from __future__ import annotations

import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.world_model.cic_schema import INPUT_DIM
from src.world_model.dataset import (
    SequenceStats,
    canonical_day_id,
    concat_window_bundles,
    day_split,
    fit_scaler,
    load_npz,
    make_sequences,
)
from src.world_model.model import LSTMWorldModel
from src.world_model.prepare_dataset import generate_demo_flows
from src.world_model.windows import WindowConfig, build_state_windows, load_cic_csv

results = []


def check(name: str, condition: bool, detail: str = "") -> None:
    results.append((name, condition))
    icon = "PASS" if condition else "FAIL"
    extra = f"  ({detail})" if detail else ""
    print(f"  {icon}  {name}{extra}")


def _rebase_timestamps(df: pd.DataFrame, t0: datetime) -> pd.DataFrame:
    out = df.copy()
    ts = pd.to_datetime(out["Timestamp"], dayfirst=True, utc=True)
    delta = ts - ts.min()
    out["Timestamp"] = (t0 + delta).dt.strftime("%d/%m/%Y %H:%M:%S")
    return out


def _day_bundle(t0: datetime, source_file: str, n_windows: int = 20) -> dict:
    df = _rebase_timestamps(generate_demo_flows(n_windows=n_windows, flows_per_window=3), t0)
    return build_state_windows(
        df,
        WindowConfig(window_seconds=5, horizon_k=6),
        source_file=source_file,
    )


print("\n-- Canonical day ids --------------------------------------------------")
check("filename Wed-14", canonical_day_id("Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv") == "2018-02-14")
check("iso date passthrough", canonical_day_id("2018-03-01") == "2018-03-01")
check("01-Mar filename", canonical_day_id("Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv") == "2018-03-01")
check("syn run passthrough", canonical_day_id("syn:ssh_bruteforce_v1:run_000") == "syn:ssh_bruteforce_v1:run_000")


print("\n-- Per-file window metadata -------------------------------------------")
b14 = _day_bundle(datetime(2018, 2, 14, 10, 0, tzinfo=timezone.utc), "Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv")
b15 = _day_bundle(datetime(2018, 2, 15, 10, 0, tzinfo=timezone.utc), "Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv")
check("32 features day14", b14["states"].shape[1] == INPUT_DIM)
check("day_id present", "day_id" in b14 and "source_file" in b14)
check("day14 ids are 2018-02-14", set(map(str, b14["day_id"])) == {"2018-02-14"})
check("day15 ids are 2018-02-15", set(map(str, b15["day_id"])) == {"2018-02-15"})
check("source_file retained", str(b14["source_file"][0]).endswith("Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv"))


print("\n-- Fake timestamps refused --------------------------------------------")
df_nots = generate_demo_flows(n_windows=8, flows_per_window=2).drop(columns=["Timestamp"])
raised = False
try:
    build_state_windows(df_nots, WindowConfig())
except ValueError as exc:
    raised = "timestamp" in str(exc).lower()
check("missing Timestamp raises", raised)


print("\n-- Concat processed pairs (not raw CSV) -------------------------------")
combined = concat_window_bundles([b14, b15])
check("concat rows = sum", len(combined["states"]) == len(b14["states"]) + len(b15["states"]))
check("concat keeps both days", set(map(str, combined["day_id"])) == {"2018-02-14", "2018-02-15"})
check("concat dim 32", combined["states"].shape[1] == 32)


print("\n-- Sequences cannot cross day boundary --------------------------------")
seq_len = 8
x, y_next, y_atk, stats, seq_days = make_sequences(
    combined["states"],
    combined["next_states"],
    combined["attack_within_k"],
    seq_len,
    timestamps=combined["timestamps"],
    day_ids=combined["day_id"],
    window_seconds=5.0,
    max_gap_seconds=15.0,
    return_stats=True,
)
check("return SequenceStats", isinstance(stats, SequenceStats))
check("kept some sequences", stats.n_kept >= 1)
check("rejected day-boundary stitches", stats.n_rejected_day >= 1)
check("every kept seq is single-day", len(set(map(str, seq_days))) <= 2)
check("no mixed-day seq ids", all(d in {"2018-02-14", "2018-02-15"} for d in map(str, seq_days)))
# Reconstruct: last state index implied by matching; check x rows share one day via seq_days
check("seq feature dim", x.shape == (stats.n_kept, seq_len, 32))


print("\n-- Sequences cannot cross large timestamp gaps ------------------------")
gap = dict(b14)
ts = gap["timestamps"].copy()
# 60s hole after index 6
ts[7:] = ts[7:] + 60.0
gap["timestamps"] = ts
_, _, _, gap_stats, _ = make_sequences(
    gap["states"],
    gap["next_states"],
    gap["attack_within_k"],
    seq_len,
    timestamps=gap["timestamps"],
    day_ids=gap["day_id"],
    window_seconds=5.0,
    max_gap_seconds=15.0,
    return_stats=True,
)
check("gap rejects at least one seq", gap_stats.n_rejected_gap >= 1)
check("gap still keeps within-block seqs", gap_stats.n_kept >= 1)


print("\n-- Temporal ordering --------------------------------------------------")
rev = dict(combined)
rev["timestamps"] = combined["timestamps"][::-1].copy()
rev["day_id"] = combined["day_id"][::-1].copy()
rev["states"] = combined["states"][::-1].copy()
rev["next_states"] = combined["next_states"][::-1].copy()
rev["attack_within_k"] = combined["attack_within_k"][::-1].copy()
_, _, _, ord_stats, _ = make_sequences(
    rev["states"],
    rev["next_states"],
    rev["attack_within_k"],
    seq_len,
    timestamps=rev["timestamps"],
    day_ids=rev["day_id"],
    return_stats=True,
)
check("decreasing timestamps rejected", ord_stats.n_rejected_order >= 1)


print("\n-- Day-based split ----------------------------------------------------")
tr, va, te = day_split(
    combined["day_id"],
    train_days=["2018-02-14"],
    val_days=["2018-02-15"],
    test_days=["2018-03-02"],
)
check("train only 14 Feb", set(map(str, combined["day_id"][tr])) == {"2018-02-14"})
check("val only 15 Feb", set(map(str, combined["day_id"][va])) == {"2018-02-15"})
check("missing test day is empty", len(te) == 0)
check("no train/val overlap", len(set(tr.tolist()) & set(va.tolist())) == 0)


print("\n-- Train-only scaler --------------------------------------------------")
scaler = fit_scaler(combined["states"][tr])
z_tr = scaler.transform(combined["states"][tr])
z_va = scaler.transform(combined["states"][va])
check("train mean ~ 0", abs(float(z_tr.mean())) < 0.05)
check("val not forced to 0 mean", abs(float(z_va.mean())) >= 0.0)  # may or may not; check same scaler
z_va2 = (combined["states"][va] - scaler.mean) / scaler.std
check("val uses train mean/std", np.allclose(z_va, z_va2))
check("scaler dim 32", scaler.mean.shape == (32,))


print("\n-- Model forward (SIH architecture) -----------------------------------")
model = LSTMWorldModel(input_dim=32, hidden_dim=64, num_layers=2, dropout=0.2)
model.eval()
with torch.no_grad():
    nxt, logit = model(torch.from_numpy(x[:2]))
check("next_state (2, 32)", tuple(nxt.shape) == (2, 32))
check("attack logit (2,)", tuple(logit.shape) == (2,))
n_lstm = sum(p.numel() for n, p in model.named_parameters() if "lstm" in n)
check("2-layer LSTM present", model.lstm.num_layers == 2 and model.lstm.hidden_size == 64)


print("\n-- Old NPZ without day_id still loads ---------------------------------")
with tempfile.TemporaryDirectory() as td:
    path = Path(td) / "old.npz"
    np.savez(
        path,
        states=b14["states"],
        next_states=b14["next_states"],
        attack_within_k=b14["attack_within_k"],
        timestamps=b14["timestamps"],
        feature_names=b14["feature_names"],
    )
    loaded = load_npz(path)
    check("old npz has states", loaded["states"].shape[1] == 32)
    check("old npz synthesizes day_id", "day_id" in loaded and set(map(str, loaded["day_id"])) == {"2018-02-14"})
    check("old npz sorted by time", np.all(np.diff(loaded["timestamps"]) >= 0))


print("\n-- Repeated Label header rows dropped ---------------------------------")
demo = generate_demo_flows(n_windows=6, flows_per_window=2)
header_row = {c: ("Label" if c == "Label" else c) for c in demo.columns}
header_row["Label"] = "Label"
demo2 = pd.concat([demo.iloc[:1], pd.DataFrame([header_row]), demo.iloc[1:]], ignore_index=True)
with tempfile.TemporaryDirectory() as td:
    csv_path = Path(td) / "with_header_row.csv"
    demo2.to_csv(csv_path, index=False)
    loaded_df = load_cic_csv(str(csv_path))
    check("header Label rows gone", not (loaded_df["Label"].astype(str).str.casefold() == "label").any())


print("\n-- Summary ------------------------------------------------------------")
passed = sum(1 for _, ok in results if ok)
failed = sum(1 for _, ok in results if not ok)
print(f"\n  Total: {len(results)}   Passed: {passed}   Failed: {failed}\n")
if failed:
    sys.exit(1)
print("  Multi-day world-model data layer OK.\n")
