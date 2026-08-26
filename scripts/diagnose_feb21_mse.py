"""Diagnosis only: 2018-02-21 next-state MSE blow-up. Does not train or mutate data."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.world_model.cic_schema import STATE_FEATURE_ORDER, bind_columns
from src.world_model.dataset import (
    DEFAULT_TEST_DAYS,
    DEFAULT_TRAIN_DAYS,
    DEFAULT_VAL_DAYS,
    Scaler,
    canonical_day_id,
    day_split,
    load_npz,
    make_sequences,
)
from src.world_model.model import LSTMWorldModel
from src.world_model.windows import load_cic_csv, parse_timestamps

NPZ = ROOT / "data" / "processed" / "state_windows_multiday.npz"
CKPT = ROOT / "src" / "world_model" / "models" / "world_lstm.pt"
SCALER = ROOT / "src" / "world_model" / "models" / "scaler.npz"
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed" / "diagnose_feb21_mse.json"

FEATURES = list(STATE_FEATURE_ORDER)
DAYS = [
    "2018-02-14",
    "2018-02-15",
    "2018-02-16",
    "2018-02-21",
    "2018-02-22",
    "2018-02-23",
    "2018-02-28",
    "2018-03-01",
    "2018-03-02",
]


def _fmt_ts(unix: float) -> str:
    return datetime.fromtimestamp(float(unix), tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _stats(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float64)
    finite = np.isfinite(x)
    n = int(x.size)
    n_nan = int(np.isnan(x).sum())
    n_posinf = int(np.isposinf(x).sum())
    n_neginf = int(np.isneginf(x).sum())
    xf = x[finite]
    if xf.size == 0:
        return {
            "n": n, "n_nan": n_nan, "n_posinf": n_posinf, "n_neginf": n_neginf,
            "min": None, "max": None, "mean": None, "std": None, "median": None, "p99": None,
        }
    return {
        "n": n,
        "n_nan": n_nan,
        "n_posinf": n_posinf,
        "n_neginf": n_neginf,
        "min": float(xf.min()),
        "max": float(xf.max()),
        "mean": float(xf.mean()),
        "std": float(xf.std()),
        "median": float(np.median(xf)),
        "p99": float(np.percentile(xf, 99)),
    }


def _csv_paths() -> dict[str, Path]:
    mapping = {}
    for p in RAW.rglob("*.csv"):
        try:
            mapping[canonical_day_id(p.name)] = p
        except ValueError:
            continue
    return mapping


def schema_report() -> dict:
    paths = _csv_paths()
    ref_day = "2018-02-14"
    ref_cols = list(pd.read_csv(paths[ref_day], nrows=0).columns)
    ref_bound = bind_columns(ref_cols)
    out = {"reference_day": ref_day, "n_ref_cols": len(ref_cols), "files": {}}
    for day, path in sorted(paths.items()):
        cols = list(pd.read_csv(path, nrows=0).columns)
        bound = bind_columns(cols)
        extra = [c for c in cols if c not in ref_cols]
        missing = [c for c in ref_cols if c not in cols]
        out["files"][day] = {
            "file": path.name,
            "n_cols": len(cols),
            "bound_fields": sorted(bound.keys()),
            "missing_vs_ref": missing,
            "extra_vs_ref": extra,
            "alias_mismatch": {
                k: (bound.get(k), ref_bound.get(k))
                for k in set(bound) | set(ref_bound)
                if bound.get(k) != ref_bound.get(k)
            },
        }
        # timestamp sample
        sample = pd.read_csv(path, nrows=5, low_memory=False)
        ts_col = bound.get("timestamp")
        lab_col = bound.get("label")
        out["files"][day]["ts_samples"] = sample[ts_col].astype(str).tolist() if ts_col else []
        if lab_col:
            vc = pd.read_csv(path, usecols=[lab_col], low_memory=False)[lab_col]
            # too heavy for 1M rows? usecols is ok. value_counts
            counts = vc.astype(str).str.strip().value_counts().head(12)
            out["files"][day]["label_counts_top"] = {str(k): int(v) for k, v in counts.items()}
            out["files"][day]["n_rows"] = int(len(vc))
    return out


def main() -> None:
    data = load_npz(NPZ)
    scaler = Scaler.load(SCALER)
    names = FEATURES
    states = data["states"]
    next_states = data["next_states"]
    days = np.array([str(d) for d in data["day_id"]])
    ts = data["timestamps"]
    wids = data.get("window_ids")

    report: dict = {"days_present": sorted(set(days.tolist()))}

    # --- 1 & 5 raw distributions + finite ---
    raw_by_day = {}
    finite_by_day = {}
    for day in DAYS:
        m = days == day
        if not m.any():
            continue
        st = states[m]
        finite_by_day[day] = {
            "n_pairs": int(m.sum()),
            "any_nan": bool(np.isnan(st).any()),
            "any_posinf": bool(np.isposinf(st).any()),
            "any_neginf": bool(np.isneginf(st).any()),
            "next_any_nan": bool(np.isnan(next_states[m]).any()),
            "next_any_inf": bool(np.isinf(next_states[m]).any()),
        }
        feats = {}
        for i, name in enumerate(names):
            feats[name] = _stats(st[:, i])
        raw_by_day[day] = feats
    report["finite"] = finite_by_day

    # compact raw table: per feature, per split-group
    groups = {
        "train": [d for d in DEFAULT_TRAIN_DAYS],
        "val": list(DEFAULT_VAL_DAYS),
        "test_2018-02-21": ["2018-02-21"],
        "test_2018-03-01": ["2018-03-01"],
        "test_2018-03-02": ["2018-03-02"],
    }
    raw_group = {}
    for gname, gdays in groups.items():
        m = np.isin(days, gdays)
        feats = {}
        for i, name in enumerate(names):
            feats[name] = _stats(states[m, i])
        raw_group[gname] = {
            "n": int(m.sum()),
            "features": feats,
        }
    report["raw_by_group"] = raw_group
    report["raw_by_day_volume"] = {
        day: {k: raw_by_day[day][k] for k in (
            "flow_count", "bytes_fwd_sum", "bytes_bwd_sum", "pkts_fwd_sum",
            "pkts_bwd_sum", "duration_mean", "duration_max", "iat_mean",
            "iat_std", "iat_max", "pkt_len_mean", "pkt_len_std", "pkt_len_max",
            "syn_flag_sum", "ack_flag_sum", "psh_flag_sum", "port_scan_score",
        )}
        for day in raw_by_day
    }

    # --- 2 scaled z-scores ---
    z = scaler.transform(states)
    z_next = scaler.transform(next_states)
    scaled = {}
    maxabs_table = {}
    for day in DAYS:
        m = days == day
        if not m.any():
            continue
        zd = z[m]
        per_feat = {}
        for i, name in enumerate(names):
            col = zd[:, i]
            per_feat[name] = {
                **_stats(col),
                "max_abs": float(np.max(np.abs(col))) if col.size else None,
            }
        scaled[day] = per_feat
        maxabs_table[day] = {name: per_feat[name]["max_abs"] for name in names}
    report["scaled_max_abs_by_day"] = maxabs_table

    # features where 21-Feb max|z| >> other days
    other_days = [d for d in DAYS if d != "2018-02-21" and d in maxabs_table]
    unusual = []
    for i, name in enumerate(names):
        z21 = maxabs_table.get("2018-02-21", {}).get(name)
        if z21 is None:
            continue
        others = [maxabs_table[d][name] for d in other_days]
        max_other = max(others) if others else 0.0
        unusual.append({
            "feature": name,
            "index": i,
            "max_abs_z_21": z21,
            "max_abs_z_other_days": float(max_other),
            "ratio_vs_other_max": float(z21 / max_other) if max_other else None,
            "train_mean": float(scaler.mean[i]),
            "train_std": float(scaler.std[i]),
        })
    unusual.sort(key=lambda r: -(r["ratio_vs_other_max"] or 0))
    report["features_unusual_z_on_21"] = unusual

    # --- 9 window timing on 21-Feb (pairs in npz, before sequences) ---
    m21 = days == "2018-02-21"
    ts21 = ts[m21]
    dts = np.diff(ts21)
    timing = {
        "n_pairs": int(m21.sum()),
        "ts_min": _fmt_ts(ts21.min()) if m21.any() else None,
        "ts_max": _fmt_ts(ts21.max()) if m21.any() else None,
        "dt_min": float(dts.min()) if dts.size else None,
        "dt_max": float(dts.max()) if dts.size else None,
        "dt_median": float(np.median(dts)) if dts.size else None,
        "dt_mean": float(dts.mean()) if dts.size else None,
        "frac_dt_eq_5": float(np.mean(np.abs(dts - 5.0) < 0.51)) if dts.size else None,
        "frac_dt_le_15": float(np.mean(dts <= 15.0 + 1e-6)) if dts.size else None,
        "n_dt_gt_15": int(np.sum(dts > 15.0 + 1e-6)) if dts.size else 0,
        "n_dt_le_0": int(np.sum(dts <= 0)) if dts.size else 0,
    }
    if wids is not None:
        w21 = wids[m21]
        dw = np.diff(w21.astype(np.int64))
        timing["window_id_diff_median"] = int(np.median(dw)) if dw.size else None
        timing["window_id_diff_max"] = int(dw.max()) if dw.size else None
        timing["frac_wid_plus_1"] = float(np.mean(dw == 1)) if dw.size else None
    report["timing_2018-02-21_pairs"] = timing

    timing_all = {}
    for day in DAYS:
        m = days == day
        if m.sum() < 2:
            continue
        dtt = np.diff(ts[m])
        timing_all[day] = {
            "n": int(m.sum()),
            "dt_median": float(np.median(dtt)),
            "dt_mean": float(dtt.mean()),
            "frac_dt_eq_5": float(np.mean(np.abs(dtt - 5.0) < 0.51)),
            "n_dt_gt_15": int(np.sum(dtt > 15.0 + 1e-6)),
            "span_hours": float((ts[m].max() - ts[m].min()) / 3600.0),
        }
    report["timing_all_days"] = timing_all

    # --- 10 sequences on 21-Feb ---
    x, y_next, y_atk, stats, seq_days = make_sequences(
        z[m21],
        z_next[m21],
        data["attack_within_k"][m21],
        8,
        timestamps=ts21,
        day_ids=days[m21],
        return_stats=True,
    )
    report["seq_2018-02-21"] = {
        "n_kept": stats.n_kept,
        "n_rejected_day": stats.n_rejected_day,
        "n_rejected_gap": stats.n_rejected_gap,
        "n_rejected_order": stats.n_rejected_order,
        "shape": list(x.shape),
        "unique_seq_days": sorted(set(map(str, seq_days))),
    }
    # reconstruct last-step timestamps for kept sequences (same loop as make_sequences)
    seq_ts = []
    seq_idx = []
    n = m21.sum()
    ts_local = ts21
    days_local = days[m21]
    kept_i = []
    for i in range(7, n):
        sl = slice(i - 7, i + 1)
        win_days = days_local[sl]
        win_ts = ts_local[sl]
        if np.any(win_days != win_days[0]):
            continue
        dtsw = np.diff(win_ts)
        if np.any(dtsw <= 0) or np.any(dtsw > 15.0 + 1e-6):
            continue
        kept_i.append(i)
        seq_ts.append(float(win_ts[-1]))
        seq_idx.append(i)
    report["seq_2018-02-21"]["n_reconstructed"] = len(kept_i)
    assert len(kept_i) == stats.n_kept

    # --- 3 & 4 model errors ---
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    model = LSTMWorldModel(input_dim=32, hidden_dim=64, num_layers=2, dropout=0.2)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(x), 256):
            batch = torch.from_numpy(x[start:start + 256])
            nxt, _ = model(batch)
            preds.append(nxt.cpu().numpy())
    pred = np.concatenate(preds, axis=0)
    persist = x[:, -1, :]
    err = pred - y_next
    persist_err = persist - y_next
    mse_row = np.mean(err ** 2, axis=1)
    persist_mse_row = np.mean(persist_err ** 2, axis=1)
    abs_err = np.abs(err)
    abs_persist = np.abs(persist_err)

    report["error_2018-02-21"] = {
        "lstm_mse": float(np.mean(err ** 2)),
        "persist_mse": float(np.mean(persist_err ** 2)),
        "lstm_mae": float(np.mean(abs_err)),
        "persist_mae": float(np.mean(abs_persist)),
        "pred_max_abs": float(np.max(np.abs(pred))),
        "target_max_abs": float(np.max(np.abs(y_next))),
        "last_state_max_abs": float(np.max(np.abs(persist))),
        "n_pred_abs_gt_50": int(np.sum(np.abs(pred) > 50)),
        "n_target_abs_gt_50": int(np.sum(np.abs(y_next) > 50)),
        "n_pred_abs_gt_20": int(np.sum(np.abs(pred) > 20)),
        "n_target_abs_gt_20": int(np.sum(np.abs(y_next) > 20)),
        "frac_lstm_worse_than_persist": float(np.mean(mse_row > persist_mse_row)),
        "median_lstm_mse": float(np.median(mse_row)),
        "median_persist_mse": float(np.median(persist_mse_row)),
        "p99_lstm_mse": float(np.percentile(mse_row, 99)),
        "p99_persist_mse": float(np.percentile(persist_mse_row, 99)),
        "max_lstm_mse": float(mse_row.max()),
        "max_persist_mse": float(persist_mse_row.max()),
    }

    # per-feature MSE on 21
    feat_mse = []
    for i, name in enumerate(names):
        feat_mse.append({
            "feature": name,
            "index": i,
            "lstm_mse": float(np.mean(err[:, i] ** 2)),
            "persist_mse": float(np.mean(persist_err[:, i] ** 2)),
            "lstm_max_abs_err": float(np.max(abs_err[:, i])),
            "persist_max_abs_err": float(np.max(abs_persist[:, i])),
            "pred_max_abs": float(np.max(np.abs(pred[:, i]))),
            "target_max_abs": float(np.max(np.abs(y_next[:, i]))),
        })
    feat_mse.sort(key=lambda r: -r["lstm_mse"])
    report["feature_mse_2018-02-21"] = feat_mse

    # top 20 sequences by LSTM MSE
    order = np.argsort(-mse_row)[:20]
    top_seq = []
    for rank, j in enumerate(order, 1):
        i_end = kept_i[j]
        feat_i = int(np.argmax(abs_err[j]))
        true_z = y_next[j]
        pred_z = pred[j]
        last_z = persist[j]
        true_raw = scaler.inverse(true_z[None, :])[0]
        pred_raw = scaler.inverse(pred_z[None, :])[0]
        last_raw = scaler.inverse(last_z[None, :])[0]
        top_seq.append({
            "rank": rank,
            "seq_end_index_in_day": int(i_end),
            "timestamp": _fmt_ts(seq_ts[j]),
            "unix": seq_ts[j],
            "lstm_mse": float(mse_row[j]),
            "persist_mse": float(persist_mse_row[j]),
            "worst_feature": names[feat_i],
            "worst_feature_index": feat_i,
            "true_z": float(true_z[feat_i]),
            "pred_z": float(pred_z[feat_i]),
            "last_z": float(last_z[feat_i]),
            "abs_err_z": float(abs_err[j, feat_i]),
            "true_raw": float(true_raw[feat_i]),
            "pred_raw": float(pred_raw[feat_i]),
            "last_raw": float(last_raw[feat_i]),
            "pred_max_abs_z_any_feat": float(np.max(np.abs(pred_z))),
            "true_max_abs_z_any_feat": float(np.max(np.abs(true_z))),
        })
    report["top20_worst_sequences"] = top_seq

    # top 20 (seq, feature) cells
    flat = abs_err.reshape(-1)
    top_flat = np.argsort(-flat)[:20]
    top_cells = []
    nseq, nf = abs_err.shape
    for rank, k in enumerate(top_flat, 1):
        j = int(k // nf)
        fi = int(k % nf)
        true_raw = float(scaler.inverse(y_next[j][None, :])[0, fi])
        pred_raw = float(scaler.inverse(pred[j][None, :])[0, fi])
        top_cells.append({
            "rank": rank,
            "timestamp": _fmt_ts(seq_ts[j]),
            "feature": names[fi],
            "index": fi,
            "true_z": float(y_next[j, fi]),
            "pred_z": float(pred[j, fi]),
            "persist_z": float(persist[j, fi]),
            "abs_err_z": float(abs_err[j, fi]),
            "persist_abs_err_z": float(abs_persist[j, fi]),
            "true_raw": true_raw,
            "pred_raw": pred_raw,
        })
    report["top20_worst_feature_errors"] = top_cells

    # compare 03-01 LSTM error for context
    for other in ("2018-03-01", "2018-03-02"):
        m = days == other
        xo, yo, _, st_o, _ = make_sequences(
            z[m], z_next[m], data["attack_within_k"][m], 8,
            timestamps=ts[m], day_ids=days[m], return_stats=True,
        )
        po = []
        with torch.no_grad():
            for start in range(0, len(xo), 256):
                nxt, _ = model(torch.from_numpy(xo[start:start + 256]))
                po.append(nxt.cpu().numpy())
        po = np.concatenate(po)
        report[f"error_{other}"] = {
            "lstm_mse": float(np.mean((po - yo) ** 2)),
            "persist_mse": float(np.mean((xo[:, -1, :] - yo) ** 2)),
            "pred_max_abs": float(np.max(np.abs(po))),
            "target_max_abs": float(np.max(np.abs(yo))),
            "n_kept": st_o.n_kept,
            "n_rejected_gap": st_o.n_rejected_gap,
        }

    print("=== schema ===")
    report["schema"] = schema_report()

    # preprocessing: no day-specific branches (static note)
    report["preprocessing_note"] = (
        "build_state_windows / bind_columns / parse_timestamps / make_sequences "
        "have no day-specific branches. 21-Feb is processed identically."
    )

    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {OUT}")

    # human summary to stdout
    print("\n=== FINITE ===")
    for day, rec in finite_by_day.items():
        print(f"  {day} n={rec['n_pairs']} nan={rec['any_nan']} inf={rec['any_posinf'] or rec['any_neginf']}")

    print("\n=== VOLUME (raw flow_count / bytes_fwd) ===")
    for day in DAYS:
        if day not in raw_by_day:
            continue
        fc = raw_by_day[day]["flow_count"]
        bf = raw_by_day[day]["bytes_fwd_sum"]
        print(
            f"  {day} n={fc['n']} flow_count max={fc['max']:.0f} p99={fc['p99']:.0f} mean={fc['mean']:.1f} | "
            f"bytes_fwd max={bf['max']:.3e} p99={bf['p99']:.3e} mean={bf['mean']:.3e}"
        )

    print("\n=== TOP unusual |z| on 21 vs other days ===")
    for row in unusual[:12]:
        print(
            f"  {row['feature']:22s} z21={row['max_abs_z_21']:10.1f} "
            f"other={row['max_abs_z_other_days']:8.1f} ratio={row['ratio_vs_other_max']}"
        )

    print("\n=== 21-Feb errors ===")
    e = report["error_2018-02-21"]
    print(json.dumps(e, indent=2))
    print("\n=== top feature MSE 21 ===")
    for row in feat_mse[:10]:
        print(
            f"  {row['feature']:22s} lstm_mse={row['lstm_mse']:12.1f} persist={row['persist_mse']:10.1f} "
            f"pred_max={row['pred_max_abs']:.1f} tgt_max={row['target_max_abs']:.1f}"
        )
    print("\n=== top 10 sequences ===")
    for row in top_seq[:10]:
        print(
            f"  {row['timestamp']} mse={row['lstm_mse']:.1f} persist={row['persist_mse']:.1f} "
            f"feat={row['worst_feature']} true_z={row['true_z']:.1f} pred_z={row['pred_z']:.1f} "
            f"true_raw={row['true_raw']:.3e} pred_raw={row['pred_raw']:.3e}"
        )
    print("\n=== timing 21 ===")
    print(json.dumps(timing, indent=2))
    print("=== seq 21 ===")
    print(json.dumps(report["seq_2018-02-21"], indent=2))


if __name__ == "__main__":
    main()
