"""
Dump why-explanations for held-out windows using the frozen LSTM.

Does not train. Loads world_lstm.pt + train-only scaler + train-only stage decoder.

Usage (from Normnative-/):
    python -m src.world_model.eval_explain --npz data/processed/state_windows_multiday.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.world_model.dataset import (
    DEFAULT_TEST_DAYS,
    DEFAULT_TRAIN_DAYS,
    Scaler,
    day_split,
    load_npz,
)
from src.world_model.eval_kstep import _load_model
from src.world_model.explain import LSTM_ATTACK_THRESHOLD, explain_forecast
from src.world_model.labels import STAGE_NAMES
from src.world_model.mitre_decode import fit_stage_decoder
from src.world_model.rollout import collect_kstep_windows
from src.world_model.train import ROOT

DEFAULT_NPZ = ROOT / "data" / "processed" / "state_windows_multiday.npz"
DEFAULT_CKPT = Path(__file__).resolve().parent / "models" / "world_lstm.pt"
DEFAULT_SCALER = Path(__file__).resolve().parent / "models" / "scaler.npz"
FOCUS_DAY = "2018-03-01"


def _jsonable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_jsonable(v) for v in obj]
    return obj


def main() -> None:
    parser = argparse.ArgumentParser(description="Why-explanations for frozen world-model forecasts")
    parser.add_argument("--npz", default=str(DEFAULT_NPZ))
    parser.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    parser.add_argument("--scaler", default=str(DEFAULT_SCALER))
    parser.add_argument("--out", default=None)
    parser.add_argument("--day", default=FOCUS_DAY)
    parser.add_argument("--n", type=int, default=8, help="How many alert windows to dump")
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--seq-len", type=int, default=8)
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
    out_path = Path(args.out) if args.out else (ckpt_path.parent / "explain_examples.json")
    if not out_path.is_absolute():
        out_path = ROOT / out_path

    data = load_npz(npz_path)
    scaler = Scaler.load(scaler_path)
    model, ckpt = _load_model(ckpt_path)
    if int(ckpt.get("input_dim", 32)) != 32 or int(ckpt.get("num_layers", 2)) != 2:
        raise SystemExit("Checkpoint architecture does not match the SIH LSTM (32-d, 2x64).")

    train_days = list(DEFAULT_TRAIN_DAYS)
    test_days = list(DEFAULT_TEST_DAYS)
    train_i, _, test_i = day_split(data["day_id"], train_days, ["2018-02-23"], test_days)
    decoder = fit_stage_decoder(
        scaler.transform(data["states"][train_i]).astype(np.float32),
        data["stage_id"][train_i],
    )

    st = scaler.transform(data["states"][test_i]).astype(np.float32)
    nxt = scaler.transform(data["next_states"][test_i]).astype(np.float32)
    batch = collect_kstep_windows(
        st, nxt, data["timestamps"][test_i], data["day_id"][test_i],
        seq_len=args.seq_len, k=args.k,
        stage_ids=data["stage_id"][test_i],
    )
    day_mask = np.array([str(d) == args.day for d in batch["day_id"]])
    idx = np.flatnonzero(day_mask)
    if len(idx) == 0:
        raise SystemExit(f"No k-step windows for day {args.day}")

    examples = []
    n_alert = 0
    n_scanned = 0
    for i in idx:
        n_scanned += 1
        bundle = explain_forecast(
            model, decoder, batch["x"][i], k=args.k, attack_threshold=LSTM_ATTACK_THRESHOLD,
        )
        true_stage = int(batch["y_stage"][i, -1]) if "y_stage" in batch else None
        bundle["true_stage_at_k"] = None if true_stage is None else STAGE_NAMES[true_stage]
        bundle["day_id"] = str(batch["day_id"][i])
        if bundle["something_bad"] or bundle["stage"] != "benign":
            examples.append(_jsonable(bundle))
            n_alert += 1
            print(f"\n[{len(examples)}] {bundle['narrative']}")
            if len(examples) >= args.n:
                break

    report = {
        "day": args.day,
        "k": args.k,
        "attack_threshold": LSTM_ATTACK_THRESHOLD,
        "n_scanned": n_scanned,
        "n_dumped": len(examples),
        "note": (
            "Frozen LSTM + train-only stage decoder. "
            "why_attack = input saliency of the attack head; "
            "why_stage = LogReg coef * Ŝ_t+k; "
            "why_change = Ŝ_t+k − S_t. No retraining."
        ),
        "examples": examples,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n[scanned] {n_scanned}  dumped={len(examples)}  alerts_or_nonbenign={n_alert}")
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
