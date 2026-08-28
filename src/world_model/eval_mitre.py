"""
Decode closed-loop Ŝ_{t+k} into SIH MITRE stages.

Does not train the LSTM. Fits a train-only multinomial LogReg on scaled
32-d states, then applies it to LSTM rollouts, persist(S_t), and true S_{t+k}.

Usage (from Normnative-/):
    python -m src.world_model.eval_mitre --npz data/processed/state_windows_multiday.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.world_model.dataset import (
    Scaler,
    load_npz,
)
from src.world_model.eval_kstep import _load_model, _split_indices
from src.world_model.labels import STAGE_NAMES
from src.world_model.mitre_decode import (
    compare_stage_rollout,
    fit_stage_decoder,
    stage_metrics,
)
from src.world_model.rollout import (
    DEFAULT_K,
    collect_kstep_windows,
    persist_rollout,
    rollout_closed_loop,
)
from src.world_model.train import ROOT, _parse_days

DEFAULT_NPZ = ROOT / "data" / "processed" / "state_windows_multiday.npz"
DEFAULT_CKPT = Path(__file__).resolve().parent / "models" / "world_lstm.pt"
DEFAULT_SCALER = Path(__file__).resolve().parent / "models" / "scaler.npz"


def _eval_split(
    name: str,
    idx: np.ndarray,
    data: dict,
    scaler: Scaler,
    model,
    decoder,
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
        stage_ids=data["stage_id"][idx],
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
    overall = compare_stage_rollout(
        batch["y_stage"], lstm_pred, persist_pred, batch["y_true"], decoder,
    )
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
        by_day[day] = compare_stage_rollout(
            batch["y_stage"][mask],
            lstm_pred[mask],
            persist_pred[mask],
            batch["y_true"][mask],
            decoder,
        )
    overall["by_day"] = by_day
    return overall


def _print_block(title: str, block: dict) -> None:
    print(f"\n=== {title}  n={block['n']} ===")
    print(
        f"{'k':>3s}  {'LSTM_acc':>9s}  {'Pers_acc':>9s}  {'Ora_acc':>8s}  "
        f"{'LSTM_mF1':>9s}  {'Pers_mF1':>9s}  {'Ora_mF1':>8s}"
    )
    for row in block["by_k"]:
        print(
            f"{row['k']:3d}  {row['lstm']['accuracy']:9.3f}  {row['persist']['accuracy']:9.3f}  "
            f"{row['oracle']['accuracy']:8.3f}  {row['lstm']['macro_f1']:9.3f}  "
            f"{row['persist']['macro_f1']:9.3f}  {row['oracle']['macro_f1']:8.3f}"
        )
    print(
        f"{'all':>3s}  {block['lstm']['accuracy']:9.3f}  {block['persist']['accuracy']:9.3f}  "
        f"{block['oracle']['accuracy']:8.3f}  {block['lstm']['macro_f1']:9.3f}  "
        f"{block['persist']['macro_f1']:9.3f}  {block['oracle']['macro_f1']:8.3f}"
    )
    print("  per-stage support / LSTM F1 / persist F1:")
    for name in STAGE_NAMES:
        a = block["lstm"]["per_stage"][name]
        b = block["persist"]["per_stage"][name]
        if a["support"] == 0 and a["pred_count"] == 0 and b["pred_count"] == 0:
            continue
        print(
            f"    {name:22s}  support={a['support']:5d}  "
            f"LSTM_f1={a['f1']:.3f}  persist_f1={b['f1']:.3f}  "
            f"tech={a['technique_id'] or '-'}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="MITRE stage decode of K-step rollouts (no LSTM training)")
    parser.add_argument("--npz", default=str(DEFAULT_NPZ))
    parser.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    parser.add_argument("--scaler", default=str(DEFAULT_SCALER))
    parser.add_argument("--out", default=None, help="JSON path (default: models/mitre_metrics.json)")
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
    out_path = Path(args.out) if args.out else (ckpt_path.parent / "mitre_metrics.json")
    if not out_path.is_absolute():
        out_path = ROOT / out_path

    print(f"[data]    {npz_path}")
    print(f"[ckpt]    {ckpt_path}")
    print(f"[scaler]  {scaler_path}")
    print(f"[device]  k={args.k}  seq_len={args.seq_len}  (LSTM frozen)")

    data = load_npz(npz_path)
    if "stage_id" not in data:
        raise SystemExit("NPZ has no stage_id — rebuild with prepare_dataset.")
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
        from src.world_model.eval_kstep import _filter_indices
        val_i = _filter_indices(data, val_i, args.source, None)
        test_i = _filter_indices(data, test_i, args.source, None)
        print(f"[source]  eval slices restricted to {args.source}")

    train_states = scaler.transform(data["states"][train_i]).astype(np.float32)
    train_stages = data["stage_id"][train_i]
    counts = {STAGE_NAMES[i]: int((train_stages == i).sum()) for i in range(len(STAGE_NAMES))}
    print(f"[decoder] train windows={len(train_i)}  stage counts={counts}")
    decoder = fit_stage_decoder(train_states, train_stages)
    seen = sorted(int(c) for c in decoder.clf.classes_)
    print(f"[decoder] classes in train: {[STAGE_NAMES[i] for i in seen]}")
    missing = [STAGE_NAMES[i] for i in range(len(STAGE_NAMES)) if i not in set(seen)]
    if missing:
        print(f"[decoder] never seen in train (cannot predict): {missing}")

    train_fit = stage_metrics(train_stages, decoder.predict(train_states))
    print(f"[decoder] train acc={train_fit['accuracy']:.3f}  macro_f1={train_fit['macro_f1']:.3f}")

    report = {
        "k": args.k,
        "seq_len": args.seq_len,
        "npz": str(npz_path),
        "ckpt": str(ckpt_path),
        "split": meta,
        "train_stage_counts": counts,
        "train_decoder": train_fit,
        "decoder_classes": [STAGE_NAMES[i] for i in seen],
        "unseen_in_train": missing,
        "note": (
            "Multinomial LogReg on scaled 32-d S_t, fit on train days only. "
            "Applied to closed-loop LSTM Ŝ_{t+k}, persist(S_t), and true S_{t+k} (oracle). "
            "LSTM weights were not updated."
        ),
    }

    val_block = _eval_split(
        "val", val_i, data, scaler, model, decoder, seq_len=args.seq_len, k=args.k,
    )
    test_block = _eval_split(
        "test", test_i, data, scaler, model, decoder, seq_len=args.seq_len, k=args.k,
    )
    if val_block and val_block.get("n"):
        _print_block("VAL MITRE stages (LSTM rollout vs persist vs oracle)", val_block)
        for day, sub in (val_block.get("by_day") or {}).items():
            _print_block(f"VAL {day}", sub)
    if test_block and test_block.get("n"):
        _print_block("TEST MITRE stages (LSTM rollout vs persist vs oracle)", test_block)
        for day, sub in (test_block.get("by_day") or {}).items():
            _print_block(f"TEST {day}", sub)

    report["val"] = val_block
    report["test"] = test_block
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
