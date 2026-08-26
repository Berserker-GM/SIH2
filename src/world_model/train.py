"""
Train the LSTM world model and a logistic-regression baseline.

Usage (from Normnative-/):
    python -m src.world_model.train --npz data/processed/state_windows.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader, TensorDataset

from src.world_model.cic_schema import INPUT_DIM
from src.world_model.dataset import (
    DEFAULT_TEST_DAYS,
    DEFAULT_TRAIN_DAYS,
    DEFAULT_VAL_DAYS,
    SequenceStats,
    canonical_day_id,
    day_split,
    fit_scaler,
    load_npz,
    make_sequences,
    temporal_split,
)
from src.world_model.metrics import (
    best_f1_threshold,
    classification_metrics,
    next_state_metrics,
)
from src.world_model.model import LSTMWorldModel

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_NPZ = ROOT / "data" / "processed" / "state_windows.npz"
DEFAULT_OUT = Path(__file__).resolve().parent / "models"


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _parse_days(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    return [canonical_day_id(x.strip()) for x in raw.split(",") if x.strip()]


def _missing_days(requested: list[str], present: set[str]) -> list[str]:
    return [d for d in requested if d not in present]


def _prepare(
    npz_path: Path,
    seq_len: int,
    *,
    train_days: list[str] | None,
    val_days: list[str] | None,
    test_days: list[str] | None,
):
    data = load_npz(npz_path)
    n = len(data["states"])
    present = sorted(set(map(str, data["day_id"])))
    present_set = set(present)
    multi_day = len(present_set) > 1
    use_day_split = multi_day or train_days is not None or val_days is not None or test_days is not None

    split_meta: dict = {
        "n_total": n,
        "days_present": present,
        "split_mode": "day" if use_day_split else "temporal",
        "missing_train": [],
        "missing_val": [],
        "missing_test": [],
    }

    if use_day_split:
        tr_days = train_days or list(DEFAULT_TRAIN_DAYS)
        va_days = val_days or list(DEFAULT_VAL_DAYS)
        te_days = test_days or list(DEFAULT_TEST_DAYS)
        train_i, val_i, test_i = day_split(data["day_id"], tr_days, va_days, te_days)
        split_meta.update({
            "train_days": tr_days,
            "val_days": va_days,
            "test_days": te_days,
            "missing_train": _missing_days(tr_days, present_set),
            "missing_val": _missing_days(va_days, present_set),
            "missing_test": _missing_days(te_days, present_set),
        })
    else:
        train_i, val_i, test_i = temporal_split(n)
        split_meta.update({
            "train_days": present,
            "val_days": present,
            "test_days": present,
        })

    split_meta["n_train"] = int(len(train_i))
    split_meta["n_val"] = int(len(val_i))
    split_meta["n_test"] = int(len(test_i))

    if len(train_i) == 0:
        raise ValueError(
            "Train split is empty. Present days: "
            f"{present}. Missing train days: {split_meta['missing_train']}"
        )

    scaler = fit_scaler(data["states"][train_i])
    seq_stats_total = SequenceStats()

    def pack(idx):
        if len(idx) == 0:
            return {
                "x": np.zeros((0, seq_len, INPUT_DIM), dtype=np.float32),
                "y_next": np.zeros((0, INPUT_DIM), dtype=np.float32),
                "y_atk": np.zeros((0,), dtype=np.float32),
                "last": np.zeros((0, INPUT_DIM), dtype=np.float32),
                "persist": np.zeros((0, INPUT_DIM), dtype=np.float32),
                "seq_days": np.array([], dtype="U10"),
                "seq_stats": SequenceStats(),
                "n_raw": 0,
                "pos_rate": 0.0,
            }
        st = scaler.transform(data["states"][idx])
        nxt = scaler.transform(data["next_states"][idx])
        y = data["attack_within_k"][idx]
        ts = data["timestamps"][idx]
        days = data["day_id"][idx]
        x_seq, y_next, y_atk, stats, seq_days = make_sequences(
            st,
            nxt,
            y,
            seq_len,
            timestamps=ts,
            day_ids=days,
            window_seconds=5.0,
            max_gap_seconds=15.0,
            return_stats=True,
        )
        seq_stats_total.n_kept += stats.n_kept
        seq_stats_total.n_rejected_day += stats.n_rejected_day
        seq_stats_total.n_rejected_gap += stats.n_rejected_gap
        seq_stats_total.n_rejected_order += stats.n_rejected_order
        last = x_seq[:, -1, :]
        persist = last
        return {
            "x": x_seq,
            "y_next": y_next,
            "y_atk": y_atk,
            "last": last.astype(np.float32),
            "persist": persist.astype(np.float32),
            "seq_days": seq_days,
            "seq_stats": stats,
            "n_raw": int(len(idx)),
            "pos_rate": float(y_atk.mean()) if len(y_atk) else 0.0,
        }

    train, val, test = pack(train_i), pack(val_i), pack(test_i)
    split_meta["seq_stats"] = {
        "n_kept": seq_stats_total.n_kept,
        "n_rejected_day": seq_stats_total.n_rejected_day,
        "n_rejected_gap": seq_stats_total.n_rejected_gap,
        "n_rejected_order": seq_stats_total.n_rejected_order,
    }
    return scaler, train, val, test, split_meta


def _loader(split: dict, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(
        torch.from_numpy(split["x"]),
        torch.from_numpy(split["y_next"]),
        torch.from_numpy(split["y_atk"]),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def train_lstm(
    train: dict,
    val: dict,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    hidden_dim: int,
    num_layers: int,
    attack_loss_weight: float,
    patience: int,
) -> tuple[LSTMWorldModel, dict]:
    device = _device()
    model = LSTMWorldModel(
        input_dim=INPUT_DIM,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
    ).to(device)

    pos = max(float(train["y_atk"].sum()), 1.0)
    neg = max(len(train["y_atk"]) - pos, 1.0)
    raw_w = neg / pos
    pos_weight = torch.tensor([min(raw_w, 3.0)], dtype=torch.float32, device=device)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    train_loader = _loader(train, batch_size, shuffle=True)
    val_loader = _loader(val, batch_size, shuffle=False)

    best_state = None
    best_val = float("inf")
    stale = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        n_seen = 0
        for xb, y_next, y_atk in train_loader:
            xb = xb.to(device)
            y_next = y_next.to(device)
            y_atk = y_atk.to(device)
            opt.zero_grad()
            pred_next, logit = model(xb)
            loss = mse(pred_next, y_next) + attack_loss_weight * bce(logit, y_atk)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_loss += loss.item() * len(xb)
            n_seen += len(xb)
        train_loss /= max(n_seen, 1)

        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for xb, y_next, y_atk in val_loader:
                xb = xb.to(device)
                y_next = y_next.to(device)
                y_atk = y_atk.to(device)
                pred_next, logit = model(xb)
                loss = mse(pred_next, y_next) + attack_loss_weight * bce(logit, y_atk)
                val_loss += loss.item() * len(xb)
                n_val += len(xb)
        val_loss /= max(n_val, 1)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"  epoch {epoch:3d}/{epochs}  train={train_loss:.4f}  val={val_loss:.4f}")

        if val_loss < best_val - 1e-4:
            best_val = val_loss
            stale = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                print(f"  early stop at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {
        "best_val_loss": best_val,
        "history": history,
        "pos_weight": float(pos_weight.item()),
    }


@torch.no_grad()
def predict_lstm(model: LSTMWorldModel, split: dict, batch_size: int = 256):
    device = next(model.parameters()).device
    model.eval()
    next_preds, probs = [], []
    loader = _loader(split, batch_size, shuffle=False)
    for xb, _, _ in loader:
        pred_next, logit = model(xb.to(device))
        next_preds.append(pred_next.cpu().numpy())
        probs.append(torch.sigmoid(logit).cpu().numpy())
    return np.concatenate(next_preds), np.concatenate(probs)


def train_logreg(train: dict, val: dict) -> tuple[LogisticRegression, float]:
    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs",
    )
    clf.fit(train["last"], train["y_atk"].astype(int))
    val_prob = clf.predict_proba(val["last"])[:, 1]
    threshold = best_f1_threshold(val["y_atk"], val_prob)
    return clf, threshold


def evaluate(model, clf, lstm_threshold: float, lr_threshold: float, split: dict, name: str) -> dict:
    next_pred, lstm_prob = predict_lstm(model, split)
    lstm_pred = (lstm_prob >= lstm_threshold).astype(int)
    lr_prob = clf.predict_proba(split["last"])[:, 1]
    lr_pred = (lr_prob >= lr_threshold).astype(int)

    persist = next_state_metrics(split["y_next"], split["persist"])
    lstm_dyn = next_state_metrics(split["y_next"], next_pred)
    persist_mse = persist["mse"]
    reduction = float((persist_mse - lstm_dyn["mse"]) / persist_mse) if persist_mse else 0.0

    lstm_atk = classification_metrics(split["y_atk"], lstm_pred)
    lr_atk = classification_metrics(split["y_atk"], lr_pred)

    by_day = {}
    seq_days = split.get("seq_days")
    if seq_days is not None and len(seq_days) == len(split["y_atk"]):
        for day in sorted(set(map(str, seq_days))):
            mask = np.array([str(d) == day for d in seq_days])
            if not mask.any():
                continue
            by_day[day] = {
                "n": int(mask.sum()),
                "lstm_attack": classification_metrics(split["y_atk"][mask], lstm_pred[mask]),
                "logreg_attack": classification_metrics(split["y_atk"][mask], lr_pred[mask]),
                "lstm_next_state": next_state_metrics(split["y_next"][mask], next_pred[mask]),
                "persist_next_state": next_state_metrics(split["y_next"][mask], split["persist"][mask]),
            }

    return {
        "split": name,
        "n": int(len(split["y_atk"])),
        "pos_rate": float(split["y_atk"].mean()) if len(split["y_atk"]) else 0.0,
        "lstm_attack": lstm_atk,
        "logreg_attack": lr_atk,
        "lstm_next_state": lstm_dyn,
        "persist_next_state": persist,
        "lstm_mse_reduction_vs_persist": reduction,
        "lstm_threshold": lstm_threshold,
        "logreg_threshold": lr_threshold,
        "beats_logreg_f1": bool(lstm_atk["f1"] >= lr_atk["f1"]),
        "beats_persist_mse": bool(lstm_dyn["mse"] < persist["mse"]),
        "by_day": by_day,
    }


def _print_class_block(title: str, block: dict) -> None:
    print(
        f"  {title:7s}  F1={block['f1']:.3f}  P={block['precision']:.3f}  "
        f"R={block['recall']:.3f}  FPR={block['fpr']:.3f}  acc={block['accuracy']:.3f}"
    )


def _print_day_table(report_split: dict) -> None:
    by_day = report_split.get("by_day") or {}
    if not by_day:
        print("  (no per-day breakdown)")
        return
    hdr = f"{'day':12s} {'model':7s} {'P':>7s} {'R':>7s} {'F1':>7s} {'FPR':>7s} {'acc':>7s} {'MSE':>10s}"
    print(hdr)
    for day, block in by_day.items():
        rows = (
            ("LSTM", block["lstm_attack"], block["lstm_next_state"]["mse"]),
            ("LogReg", block["logreg_attack"], None),
        )
        for model_name, atk, mse in rows:
            mse_s = f"{mse:10.4f}" if mse is not None else f"{'n/a':>10s}"
            print(
                f"{day:12s} {model_name:7s} "
                f"{atk['precision']:7.3f} {atk['recall']:7.3f} {atk['f1']:7.3f} "
                f"{atk['fpr']:7.3f} {atk['accuracy']:7.3f} {mse_s}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LSTM world model vs logistic regression")
    parser.add_argument("--npz", default=str(DEFAULT_NPZ))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--attack-loss-weight", type=float, default=0.5)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument(
        "--train-days",
        default=None,
        help="Comma-separated days (YYYY-MM-DD or CIC filename). Default: 14/15/16/22/28 Feb 2018",
    )
    parser.add_argument("--val-days", default=None, help="Validation days (default: 2018-02-23)")
    parser.add_argument(
        "--test-days",
        default=None,
        help="Test days (default: 2018-02-21,2018-03-01,2018-03-02)",
    )
    parser.add_argument(
        "--check-data",
        action="store_true",
        help="Load NPZ, build splits/sequences, print leakage stats, then exit (no training)",
    )
    args = parser.parse_args()

    npz_path = Path(args.npz)
    if not npz_path.is_absolute():
        npz_path = ROOT / npz_path
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[data] {npz_path}")
    scaler, train, val, test, sizes = _prepare(
        npz_path,
        args.seq_len,
        train_days=_parse_days(args.train_days),
        val_days=_parse_days(args.val_days),
        test_days=_parse_days(args.test_days),
    )
    print(
        f"[split] mode={sizes['split_mode']}  "
        f"train={sizes['n_train']} val={sizes['n_val']} test={sizes['n_test']}  "
        f"seq_len={args.seq_len}  sequences train={len(train['x'])} val={len(val['x'])} test={len(test['x'])}"
    )
    print(f"[days] present={sizes['days_present']}")
    if sizes["split_mode"] == "day":
        print(f"       train_days={sizes['train_days']}")
        print(f"       val_days={sizes['val_days']}")
        print(f"       test_days={sizes['test_days']}")
        print(
            f"       missing train={sizes['missing_train'] or 'none'}  "
            f"val={sizes['missing_val'] or 'none'}  "
            f"test={sizes['missing_test'] or 'none'}"
        )
    ss = sizes["seq_stats"]
    print(
        f"[seq]  kept={ss['n_kept']}  rejected_day={ss['n_rejected_day']}  "
        f"rejected_gap={ss['n_rejected_gap']}  rejected_order={ss['n_rejected_order']}"
    )
    print(f"[label] train_pos={train['pos_rate']:.3f} val_pos={val['pos_rate']:.3f} test_pos={test['pos_rate']:.3f}")
    print(f"[device] {_device()}")

    blocking = []
    if sizes["split_mode"] == "day":
        if sizes["missing_val"]:
            blocking.append(f"validation days missing: {sizes['missing_val']}")
        if sizes["missing_test"]:
            blocking.append(f"test days missing: {sizes['missing_test']}")
        if len(val["x"]) == 0:
            blocking.append("validation sequences empty")
        if len(test["x"]) == 0:
            blocking.append("test sequences empty")
    if args.check_data:
        print("\n[check-data] split/sequence construction only — not training")
        if blocking:
            print("[check-data] NOT READY for full training:")
            for item in blocking:
                print(f"  - {item}")
            raise SystemExit(2)
        print("[check-data] splits are complete; full training is possible")
        return

    if blocking:
        print("\n[stop] Cannot start full training (do not fabricate a split):")
        for item in blocking:
            print(f"  - {item}")
        print("Add the missing CIC-IDS2018 CSVs and re-run prepare_dataset, or pass different --val-days/--test-days.")
        raise SystemExit(2)

    print("\n[LSTM] training world model (next-state + attack_within_k)")
    model, train_info = train_lstm(
        train,
        val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        attack_loss_weight=args.attack_loss_weight,
        patience=args.patience,
    )

    print("\n[LR] fitting logistic regression on the same 32 features")
    clf, lr_threshold = train_logreg(train, val)

    _, val_prob = predict_lstm(model, val)
    lstm_threshold = best_f1_threshold(val["y_atk"], val_prob)
    print(f"[thr] LSTM={lstm_threshold:.2f}  LogReg={lr_threshold:.2f}  (chosen on val F1)")

    report = {
        "sizes": sizes,
        "seq_len": args.seq_len,
        "input_dim": INPUT_DIM,
        "train_info": {
            "best_val_loss": train_info["best_val_loss"],
            "pos_weight": train_info["pos_weight"],
        },
        "val": evaluate(model, clf, lstm_threshold, lr_threshold, val, "val"),
        "test": evaluate(model, clf, lstm_threshold, lr_threshold, test, "test"),
    }

    weights_path = out_dir / "world_lstm.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": INPUT_DIM,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "seq_len": args.seq_len,
            "lstm_threshold": lstm_threshold,
        },
        weights_path,
    )
    scaler.save(out_dir / "scaler.npz")
    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\n[saved] {weights_path}")
    print(f"[saved] {out_dir / 'scaler.npz'}")
    print(f"[saved] {metrics_path}")

    test_m = report["test"]
    print("\n=== OVERALL TEST ===")
    print("Attack-within-K classification")
    for name, block in (("LSTM", test_m["lstm_attack"]), ("LogReg", test_m["logreg_attack"])):
        _print_class_block(name, block)
    print("Next-state dynamics (scaled; lower is better)")
    print(
        f"  LSTM     MSE={test_m['lstm_next_state']['mse']:.4f}  "
        f"MAE={test_m['lstm_next_state']['mae']:.4f}  "
        f"RMSE={test_m['lstm_next_state']['rmse']:.4f}"
    )
    print(
        f"  Persist  MSE={test_m['persist_next_state']['mse']:.4f}  "
        f"MAE={test_m['persist_next_state']['mae']:.4f}  "
        f"RMSE={test_m['persist_next_state']['rmse']:.4f}"
    )
    print(f"  LSTM MSE reduction vs persist: {test_m['lstm_mse_reduction_vs_persist']:.3f}")
    print(f"  LSTM F1 vs LogReg: {'WIN' if test_m['beats_logreg_f1'] else 'LOSE'}  "
          f"(acc {test_m['lstm_attack']['accuracy']:.3f} vs {test_m['logreg_attack']['accuracy']:.3f}, "
          f"FPR {test_m['lstm_attack']['fpr']:.3f} vs {test_m['logreg_attack']['fpr']:.3f})")
    print(f"  LSTM MSE vs persist: {'WIN' if test_m['beats_persist_mse'] else 'LOSE'}")
    print("\n=== PER DAY (test) ===")
    _print_day_table(test_m)
    if report["val"].get("by_day"):
        print("\n=== PER DAY (val) ===")
        _print_day_table(report["val"])


if __name__ == "__main__":
    main()
