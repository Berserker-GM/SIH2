"""
Evaluate closed-loop K-step forecasts from a trained LSTM world model.

Does not train. Loads world_lstm.pt + the train-only scaler.

Usage (from Normnative-/):
    python -m src.world_model.eval_kstep --npz data/processed/state_windows_multiday.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.world_model.dataset import (
    DEFAULT_TEST_DAYS,
    DEFAULT_TRAIN_DAYS,
    DEFAULT_VAL_DAYS,
    Scaler,
    canonical_day_id,
    day_split,
    load_npz,
    split_synthetic_days,
    temporal_split,
)
from src.world_model.model import LSTMWorldModel
from src.world_model.rollout import (
    DEFAULT_K,
    collect_kstep_windows,
    compare_kstep,
    persist_rollout,
    rollout_closed_loop,
)
from src.world_model.train import ROOT, _device, _parse_days

DEFAULT_NPZ = ROOT / "data" / "processed" / "state_windows_multiday.npz"
DEFAULT_CKPT = Path(__file__).resolve().parent / "models" / "world_lstm.pt"
DEFAULT_SCALER = Path(__file__).resolve().parent / "models" / "scaler.npz"


def _filter_indices(data: dict, idx: np.ndarray, source: str | None, days: list[str] | None) -> np.ndarray:
    if len(idx) == 0:
        return idx
    keep = np.ones(len(idx), dtype=bool)
    if source and source not in {"all", "*"}:
        src = np.asarray(data.get("source", np.array(["real"] * len(data["states"]))))
        keep &= src[idx] == source
    if days:
        want = {canonical_day_id(d) for d in days}
        keep &= np.array([canonical_day_id(d) in want for d in data["day_id"][idx]])
    return idx[keep]


def _load_model(ckpt_path: Path) -> tuple[LSTMWorldModel, dict]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = LSTMWorldModel(
        input_dim=int(ckpt.get("input_dim", 32)),
        hidden_dim=int(ckpt.get("hidden_dim", 64)),
        num_layers=int(ckpt.get("num_layers", 2)),
        dropout=0.2,
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(_device())
    model.eval()
    return model, ckpt


def _split_indices(data: dict, train_days, val_days, test_days):
    n = len(data["states"])
    present = sorted(set(map(str, data["day_id"])))
    present_set = set(present)
    multi = len(present_set) > 1
    use_day = multi or train_days is not None or val_days is not None or test_days is not None
    if use_day:
        tr = train_days or list(DEFAULT_TRAIN_DAYS)
        va = val_days or list(DEFAULT_VAL_DAYS)
        te = test_days or list(DEFAULT_TEST_DAYS)
        syn_tr, syn_va, syn_te = split_synthetic_days(present)
        if syn_tr or syn_va or syn_te:
            tr = list(tr) + [d for d in syn_tr if d not in tr]
            va = list(va) + [d for d in syn_va if d not in va]
            te = list(te) + [d for d in syn_te if d not in te]
        return day_split(data["day_id"], tr, va, te), {
            "mode": "day",
            "present": present,
            "train_days": tr,
            "val_days": va,
            "test_days": te,
            "synth_train_days": syn_tr,
            "synth_val_days": syn_va,
            "synth_test_days": syn_te,
        }
    return temporal_split(n), {"mode": "temporal", "present": present}


def _eval_split(
    name: str,
    idx: np.ndarray,
    data: dict,
    scaler: Scaler,
    model: LSTMWorldModel,
    *,
    seq_len: int,
    k: int,
) -> dict | None:
    if len(idx) == 0:
        print(f"[{name}] empty split — skip")
        return None
    st = scaler.transform(data["states"][idx]).astype(np.float32)
    nxt = scaler.transform(data["next_states"][idx]).astype(np.float32)
    batch = collect_kstep_windows(
        st,
        nxt,
        data["timestamps"][idx],
        data["day_id"][idx],
        seq_len=seq_len,
        k=k,
    )
    print(
        f"[{name}] k-step windows kept={batch['n_kept']}  "
        f"rejected_day={batch['n_rejected_day']}  "
        f"rejected_gap={batch['n_rejected_gap']}  "
        f"rejected_order={batch['n_rejected_order']}"
    )
    if batch["n_kept"] == 0:
        return {"split": name, "n": 0, "collect": {
            "n_kept": 0,
            "n_rejected_day": batch["n_rejected_day"],
            "n_rejected_gap": batch["n_rejected_gap"],
            "n_rejected_order": batch["n_rejected_order"],
        }}

    lstm_pred, _ = rollout_closed_loop(model, batch["x"], k=k)
    persist_pred = persist_rollout(batch["x"][:, -1, :], k=k)
    overall = compare_kstep(batch["y_true"], lstm_pred, persist_pred)
    overall["split"] = name
    overall["collect"] = {
        "n_kept": batch["n_kept"],
        "n_rejected_day": batch["n_rejected_day"],
        "n_rejected_gap": batch["n_rejected_gap"],
        "n_rejected_order": batch["n_rejected_order"],
    }

    by_day = {}
    for day in sorted(set(map(str, batch["day_id"]))):
        mask = np.array([str(d) == day for d in batch["day_id"]])
        by_day[day] = compare_kstep(
            batch["y_true"][mask],
            lstm_pred[mask],
            persist_pred[mask],
        )
    overall["by_day"] = by_day
    return overall


def _print_compare(title: str, block: dict) -> None:
    print(f"\n=== {title}  n={block['n']} ===")
    print(
        f"{'k':>3s}  {'LSTM_MSE':>10s}  {'Persist_MSE':>12s}  {'LSTM_MAE':>10s}  "
        f"{'Persist_MAE':>12s}  {'ΔMSE':>8s}"
    )
    for row in block["by_k"]:
        print(
            f"{row['k']:3d}  {row['lstm']['mse']:10.4f}  {row['persist']['mse']:12.4f}  "
            f"{row['lstm']['mae']:10.4f}  {row['persist']['mae']:12.4f}  "
            f"{row['mse_reduction_vs_persist']:8.3f}"
        )
    lo = block["lstm_overall"]
    po = block["persist_overall"]
    print(
        f"{'all':>3s}  {lo['mse']:10.4f}  {po['mse']:12.4f}  "
        f"{lo['mae']:10.4f}  {po['mae']:12.4f}  "
        f"{block['mse_reduction_vs_persist']:8.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="K-step closed-loop eval (no training)")
    parser.add_argument("--npz", default=str(DEFAULT_NPZ))
    parser.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    parser.add_argument("--scaler", default=str(DEFAULT_SCALER))
    parser.add_argument("--out", default=None, help="JSON path (default: models/kstep_metrics.json)")
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--train-days", default=None)
    parser.add_argument("--val-days", default=None)
    parser.add_argument("--test-days", default=None)
    parser.add_argument("--source", default="all", help="all | real | synthetic")
    args = parser.parse_args()

    npz_path = Path(args.npz)
    if not npz_path.is_absolute():
        npz_path = ROOT / npz_path
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.is_absolute():
        ckpt_path = ROOT / ckpt_path
    scaler_path = Path(args.scaler)
    if not scaler_path.is_absolute():
        scaler_path = ROOT / scaler_path
    out_path = Path(args.out) if args.out else (ckpt_path.parent / "kstep_metrics.json")
    if not out_path.is_absolute():
        out_path = ROOT / out_path

    print(f"[data]    {npz_path}")
    print(f"[ckpt]    {ckpt_path}")
    print(f"[scaler]  {scaler_path}")
    print(f"[device]  {_device()}  k={args.k}  seq_len={args.seq_len}")

    data = load_npz(npz_path)
    scaler = Scaler.load(scaler_path)
    model, ckpt = _load_model(ckpt_path)
    print(
        f"[model]   input={ckpt.get('input_dim')} hidden={ckpt.get('hidden_dim')} "
        f"layers={ckpt.get('num_layers')} seq_len={ckpt.get('seq_len')}"
    )
    if int(ckpt.get("input_dim", 32)) != 32 or int(ckpt.get("num_layers", 2)) != 2:
        raise SystemExit("Checkpoint architecture does not match the SIH LSTM (32-d, 2x64).")

    (train_i, val_i, test_i), meta = _split_indices(
        data,
        _parse_days(args.train_days),
        _parse_days(args.val_days),
        _parse_days(args.test_days),
    )
    print(f"[split]   mode={meta['mode']} present={meta['present']}")
    if args.source != "all":
        train_i = _filter_indices(data, train_i, args.source, None)
        val_i = _filter_indices(data, val_i, args.source, None)
        test_i = _filter_indices(data, test_i, args.source, None)
        print(f"[source]  {args.source}  train={len(train_i)} val={len(val_i)} test={len(test_i)}")

    report = {
        "k": args.k,
        "seq_len": args.seq_len,
        "npz": str(npz_path),
        "ckpt": str(ckpt_path),
        "split": meta,
        "note": "Closed-loop rollout of the saved next-state head. No retraining.",
    }

    val_block = _eval_split("val", val_i, data, scaler, model, seq_len=args.seq_len, k=args.k)
    test_block = _eval_split("test", test_i, data, scaler, model, seq_len=args.seq_len, k=args.k)
    if val_block and val_block.get("n"):
        _print_compare("VAL K-step (closed loop vs persist)", val_block)
        if val_block.get("by_day"):
            for day, sub in val_block["by_day"].items():
                _print_compare(f"VAL {day}", sub)
    if test_block and test_block.get("n"):
        _print_compare("TEST K-step (closed loop vs persist)", test_block)
        if test_block.get("by_day"):
            for day, sub in test_block["by_day"].items():
                _print_compare(f"TEST {day}", sub)

    report["val"] = val_block
    report["test"] = test_block
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
