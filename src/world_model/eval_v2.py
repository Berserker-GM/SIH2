"""v1 vs v2 comparison: scaler regression, K-step, MITRE slices, explainability.

Does not train. Expects:
  v1  world_lstm.pt + scaler.npz  (CIC-only, quoted run — never overwritten)
  v2  world_lstm_v2.pt + scaler_v2.npz  (optional; skip v2 rows if missing)

Usage (from Normnative-/):
  python -m src.world_model.eval_v2 \\
      --npz-v1 data/processed/state_windows_multiday.npz \\
      --npz-v2 data/processed/state_windows_combined.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.world_model.cic_schema import PCAP_ONLY_FEATURES, STATE_FEATURE_ORDER
from src.world_model.dataset import (
    DEFAULT_TEST_DAYS,
    DEFAULT_TRAIN_DAYS,
    Scaler,
    day_split,
    fit_scaler,
    load_npz,
    split_synthetic_days,
)
from src.world_model.eval_kstep import _eval_split as kstep_eval, _filter_indices, _load_model
from src.world_model.eval_mitre import _eval_split as mitre_eval
from src.world_model.explain import explain_forecast
from src.world_model.labels import STAGE_NAMES
from src.world_model.mitre_decode import fit_stage_decoder
from src.world_model.personas import generate_persona_fixture_frame
from src.world_model.train import ROOT

DEFAULT_NPZ_V1 = ROOT / "data" / "processed" / "state_windows_multiday.npz"
DEFAULT_NPZ_V2 = ROOT / "data" / "processed" / "state_windows_combined.npz"
DEFAULT_CKPT_V1 = ROOT / "src" / "world_model" / "models" / "world_lstm.pt"
DEFAULT_SCALER_V1 = ROOT / "src" / "world_model" / "models" / "scaler.npz"
DEFAULT_CKPT_V2 = ROOT / "src" / "world_model" / "models" / "world_lstm_v2.pt"
DEFAULT_SCALER_V2 = ROOT / "src" / "world_model" / "models" / "scaler_v2.npz"

C2_DAY = "2018-03-01"  # quoted infiltration day
BOT_DAY = "2018-03-02"  # real held-out C2
NEW_DIMS = list(PCAP_ONLY_FEATURES)


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _train_indices(data: dict) -> np.ndarray:
    present = sorted(set(map(str, data["day_id"])))
    tr = list(DEFAULT_TRAIN_DAYS)
    va = ["2018-02-23"]
    te = list(DEFAULT_TEST_DAYS)
    syn_tr, syn_va, syn_te = split_synthetic_days(present)
    tr = tr + [d for d in syn_tr if d not in tr]
    va = va + [d for d in syn_va if d not in va]
    te = te + [d for d in syn_te if d not in te]
    train_i, _, _ = day_split(data["day_id"], tr, va, te)
    return train_i


def _test_indices(data: dict) -> np.ndarray:
    present = sorted(set(map(str, data["day_id"])))
    tr = list(DEFAULT_TRAIN_DAYS)
    va = ["2018-02-23"]
    te = list(DEFAULT_TEST_DAYS)
    syn_tr, syn_va, syn_te = split_synthetic_days(present)
    te = te + [d for d in syn_te if d not in te]
    _, _, test_i = day_split(data["day_id"], tr + syn_tr, va + syn_va, te)
    return test_i


def scaler_regression(data_v1: dict, data_v2: dict, scaler_v1: Scaler) -> dict:
    """Fit combined-train scaler; compare z-shift on real test days. No LSTM training."""
    train_v2 = _train_indices(data_v2)
    combined_scaler = fit_scaler(data_v2["states"][train_v2])
    std = combined_scaler.std
    blown = [
        STATE_FEATURE_ORDER[i]
        for i, s in enumerate(std)
        if not np.isfinite(s) or s > 1e8
    ]
    dim_rows = []
    for i, name in enumerate(STATE_FEATURE_ORDER):
        row = {
            "name": name,
            "v1_mean": float(scaler_v1.mean[i]),
            "v1_std": float(scaler_v1.std[i]),
            "v2_mean": float(combined_scaler.mean[i]),
            "v2_std": float(combined_scaler.std[i]),
            "mean_shift": float(combined_scaler.mean[i] - scaler_v1.mean[i]),
            "std_ratio": float(combined_scaler.std[i] / scaler_v1.std[i]) if scaler_v1.std[i] else None,
        }
        dim_rows.append(row)

    real_test = _filter_indices(data_v1, _test_indices(data_v1), "real", None)
    z1 = scaler_v1.transform(data_v1["states"][real_test])
    z2 = combined_scaler.transform(data_v1["states"][real_test])
    delta = z2 - z1
    per_dim_rmse = np.sqrt(np.mean(delta ** 2, axis=0))
    frozen = list(range(29))  # dims 0–28 should barely move
    new = list(range(29, 32))
    report = {
        "n_combined_train": int(len(train_v2)),
        "n_real_test": int(len(real_test)),
        "blown_up_dims": blown,
        "new_dims_v2_std": {name: float(combined_scaler.std[STATE_FEATURE_ORDER.index(name)]) for name in NEW_DIMS},
        "new_dims_v1_std": {name: float(scaler_v1.std[STATE_FEATURE_ORDER.index(name)]) for name in NEW_DIMS},
        "z_rmse_dims_0_28": float(np.mean(per_dim_rmse[frozen])),
        "z_rmse_dims_29_31": float(np.mean(per_dim_rmse[new])),
        "z_rmse_max_dim": {
            "name": STATE_FEATURE_ORDER[int(np.argmax(per_dim_rmse))],
            "rmse": float(np.max(per_dim_rmse)),
        },
        "per_dim": dim_rows,
        "pass_no_blowup": len(blown) == 0,
        "pass_real_z_stable": float(np.mean(per_dim_rmse[frozen])) < 0.15,
        "combined_scaler": combined_scaler,
    }
    print("\n=== SCALER REGRESSION (combined-train scaler on real test, before v2 LSTM train) ===")
    print(f"  blown-up dims: {blown or 'none'}")
    print(f"  dims 29–31  v1 std={report['new_dims_v1_std']}  v2 std={report['new_dims_v2_std']}")
    print(f"  real-test z RMSE dims 0–28: {report['z_rmse_dims_0_28']:.4f}  (pass < 0.15: {report['pass_real_z_stable']})")
    print(f"  real-test z RMSE dims 29–31: {report['z_rmse_dims_29_31']:.4f}  (expected to move; were stubs)")
    return report


def _kstep_slice(name, idx, data, scaler, model, seq_len, k):
    block = kstep_eval(name, idx, data, scaler, model, seq_len=seq_len, k=k)
    if not block:
        return {"split": name, "n": 0}
    return block


def _mitre_slice(name, idx, data, scaler, model, decoder, seq_len, k):
    block = mitre_eval(name, idx, data, scaler, model, decoder, seq_len=seq_len, k=k)
    if not block:
        return {"split": name, "n": 0}
    return block


def _c2_recall(block: dict) -> dict:
    stage = (block.get("lstm") or {}).get("per_stage") or {}
    c2 = stage.get("command_and_control") or {}
    return {
        "support": int(c2.get("support") or 0),
        "f1": float(c2.get("f1") or 0.0),
        "recall": float(c2.get("recall") or 0.0),
        "pred_count": int(c2.get("pred_count") or 0),
    }


def explainability_check(model, decoder, scaler) -> dict:
    """Confirm dims 29–31 can surface in why_* without being a generator-only tell."""
    from datetime import datetime, timezone

    frames = {
        "recon": generate_persona_fixture_frame(
            "recon_portscan_v1", "explain", "reconnaissance", "T1595.001",
            n_windows=16, t0=datetime(2026, 7, 1, tzinfo=timezone.utc), seed=1,
        ),
        "evasive": generate_persona_fixture_frame(
            "evasive_lateral_v1", "explain", "lateral_movement", "T1021",
            n_windows=16, t0=datetime(2026, 7, 2, tzinfo=timezone.utc), seed=2,
        ),
        "c2": generate_persona_fixture_frame(
            "bot_c2_beacon_v1", "explain", "command_and_control", "T1071",
            n_windows=16, t0=datetime(2026, 7, 3, tzinfo=timezone.utc), seed=3,
        ),
        "benign": generate_persona_fixture_frame(
            "benign_office_v1", "explain", "benign", "",
            n_windows=16, t0=datetime(2026, 7, 4, tzinfo=timezone.utc), seed=4,
        ),
    }
    out = {}
    for name, df in frames.items():
        hist = df[list(STATE_FEATURE_ORDER)].to_numpy(dtype=np.float32)[-8:]
        scaled = scaler.transform(hist).astype(np.float32)
        bundle = explain_forecast(model, decoder, scaled)
        why_names = [r["name"] for r in bundle["why_attack"]]
        stage_names = [r["name"] for r in bundle["why_stage"]]
        change_names = [r["name"] for r in bundle["why_change"]]
        new_in_attack = [n for n in why_names if n in NEW_DIMS]
        out[name] = {
            "stage": bundle["stage"],
            "attack_probability": bundle["attack_probability"],
            "why_attack": why_names,
            "why_stage": stage_names,
            "why_change": change_names,
            "new_dims_in_why_attack": new_in_attack,
        }
        print(f"  [{name}] stage={bundle['stage']}  P(atk)={bundle['attack_probability']:.3f}  "
              f"why_attack={why_names[:3]}  new_dims={new_in_attack or 'none'}")

    recon_top = out["recon"]["why_attack"][0] if out["recon"]["why_attack"] else ""
    evasive_uses_new = bool(out["evasive"]["new_dims_in_why_attack"] or
                            any(n in NEW_DIMS for n in out["evasive"]["why_stage"][:3]))
    recon_not_only_ttl = recon_top != "ttl_variance"
    out["sane"] = {
        "recon_not_keyed_only_on_ttl": recon_not_only_ttl,
        "evasive_can_cite_new_dims": evasive_uses_new,
        "pass": recon_not_only_ttl,
    }
    print(f"  sanity: recon top != ttl_variance → {recon_not_only_ttl}; "
          f"evasive cites new dims → {evasive_uses_new}")
    return out


def _row(slice_name, version, kstep, mitre, extra=None):
    n = int((kstep or {}).get("n") or 0)
    mse = (kstep or {}).get("lstm_overall", {}).get("mse")
    dms = (kstep or {}).get("mse_reduction_vs_persist")
    acc = (mitre or {}).get("lstm", {}).get("accuracy")
    mf1 = (mitre or {}).get("lstm", {}).get("macro_f1")
    rec = extra or {}
    return {
        "slice": slice_name,
        "version": version,
        "n": n,
        "lstm_mse": mse,
        "mse_reduction_vs_persist": dms,
        "mitre_accuracy": acc,
        "mitre_macro_f1": mf1,
        **rec,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="v1 vs v2 scaler / K-step / MITRE / explain checks")
    parser.add_argument("--npz-v1", default=str(DEFAULT_NPZ_V1))
    parser.add_argument("--npz-v2", default=str(DEFAULT_NPZ_V2))
    parser.add_argument("--ckpt-v1", default=str(DEFAULT_CKPT_V1))
    parser.add_argument("--scaler-v1", default=str(DEFAULT_SCALER_V1))
    parser.add_argument("--ckpt-v2", default=str(DEFAULT_CKPT_V2))
    parser.add_argument("--scaler-v2", default=str(DEFAULT_SCALER_V2))
    parser.add_argument("--out", default=None)
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--skip-v2", action="store_true")
    args = parser.parse_args()

    npz_v1 = _resolve(args.npz_v1)
    npz_v2 = _resolve(args.npz_v2)
    ckpt_v1 = _resolve(args.ckpt_v1)
    scaler_v1_path = _resolve(args.scaler_v1)
    ckpt_v2 = _resolve(args.ckpt_v2)
    scaler_v2_path = _resolve(args.scaler_v2)
    out_path = _resolve(args.out) if args.out else (ROOT / "src" / "world_model" / "models" / "v2_metrics.json")

    print(f"[v1] npz={npz_v1}")
    print(f"[v1] ckpt={ckpt_v1}  scaler={scaler_v1_path}")
    data_v1 = load_npz(npz_v1)
    scaler_v1 = Scaler.load(scaler_v1_path)
    model_v1, _ = _load_model(ckpt_v1)

    data_v2 = load_npz(npz_v2) if npz_v2.is_file() else data_v1
    scal_reg = scaler_regression(data_v1, data_v2, scaler_v1)
    combined_scaler = scal_reg.pop("combined_scaler")

    real_test_v1 = _filter_indices(data_v1, _test_indices(data_v1), "real", None)
    mar01 = _filter_indices(data_v1, real_test_v1, "real", [C2_DAY])  # 01 Mar infiltration (quoted)
    mar02 = _filter_indices(data_v1, real_test_v1, "real", [BOT_DAY])

    print("\n=== K-STEP regression: v1 weights + combined scaler on real test (no synth in train of v1) ===")
    k_v1_combo = _kstep_slice("real_test_v1w_combined_scaler", real_test_v1, data_v1, combined_scaler, model_v1, args.seq_len, args.k)
    k_v1_orig = _kstep_slice("real_test_v1", real_test_v1, data_v1, scaler_v1, model_v1, args.seq_len, args.k)
    k_v1_mar01 = _kstep_slice("real_01Mar_v1", mar01, data_v1, scaler_v1, model_v1, args.seq_len, args.k)
    k_v1_mar02 = _kstep_slice("real_02Mar_v1", mar02, data_v1, scaler_v1, model_v1, args.seq_len, args.k)

    train_i_v1 = _train_indices(data_v1)
    dec_v1 = fit_stage_decoder(
        scaler_v1.transform(data_v1["states"][train_i_v1]).astype(np.float32),
        data_v1["stage_id"][train_i_v1],
    )
    m_v1_real = _mitre_slice("real_test_v1", real_test_v1, data_v1, scaler_v1, model_v1, dec_v1, args.seq_len, args.k)
    m_v1_mar02 = _mitre_slice("real_02Mar_v1", mar02, data_v1, scaler_v1, model_v1, dec_v1, args.seq_len, args.k)

    rows = [
        _row("real test days (quoted v1 scaler)", "v1", k_v1_orig, m_v1_real),
        _row("real test days (combined scaler, v1 weights)", "v1+rescaled", k_v1_combo, None),
        _row("real 01 Mar infiltration", "v1", k_v1_mar01, None),
        _row("real 02 Mar C2", "v1", k_v1_mar02, m_v1_mar02, {"c2": _c2_recall(m_v1_mar02)}),
    ]

    v2_loaded = ckpt_v2.is_file() and scaler_v2_path.is_file() and not args.skip_v2
    explain = None
    if v2_loaded:
        print(f"\n[v2] ckpt={ckpt_v2}  scaler={scaler_v2_path}")
        model_v2, _ = _load_model(ckpt_v2)
        scaler_v2 = Scaler.load(scaler_v2_path)
        train_i_v2 = _train_indices(data_v2)
        dec_v2 = fit_stage_decoder(
            scaler_v2.transform(data_v2["states"][train_i_v2]).astype(np.float32),
            data_v2["stage_id"][train_i_v2],
        )
        seen = sorted(int(c) for c in dec_v2.clf.classes_)
        print(f"[v2 decoder] classes={[STAGE_NAMES[i] for i in seen]}")

        real_test_v2 = _filter_indices(data_v2, _test_indices(data_v2), "real", None)
        synth_test = _filter_indices(data_v2, _test_indices(data_v2), "synthetic", None)
        mar02_v2 = _filter_indices(data_v2, real_test_v2, "real", [BOT_DAY])
        mar01_v2 = _filter_indices(data_v2, real_test_v2, "real", [C2_DAY])

        k_v2_real = _kstep_slice("real_test_v2", real_test_v2, data_v2, scaler_v2, model_v2, args.seq_len, args.k)
        k_v2_synth = _kstep_slice("synth_test_v2", synth_test, data_v2, scaler_v2, model_v2, args.seq_len, args.k)
        k_v2_mar01 = _kstep_slice("real_01Mar_v2", mar01_v2, data_v2, scaler_v2, model_v2, args.seq_len, args.k)
        k_v2_mar02 = _kstep_slice("real_02Mar_v2", mar02_v2, data_v2, scaler_v2, model_v2, args.seq_len, args.k)

        m_v2_real = _mitre_slice("real_test_v2", real_test_v2, data_v2, scaler_v2, model_v2, dec_v2, args.seq_len, args.k)
        m_v2_synth = _mitre_slice("synth_test_v2", synth_test, data_v2, scaler_v2, model_v2, dec_v2, args.seq_len, args.k)
        m_v2_mar02 = _mitre_slice("real_02Mar_v2", mar02_v2, data_v2, scaler_v2, model_v2, dec_v2, args.seq_len, args.k)

        rows.extend([
            _row("real test days", "v2", k_v2_real, m_v2_real),
            _row("synthetic held-out runs", "v2", k_v2_synth, m_v2_synth),
            _row("real 01 Mar infiltration", "v2", k_v2_mar01, None),
            _row("real 02 Mar C2 (transfer)", "v2", k_v2_mar02, m_v2_mar02, {"c2": _c2_recall(m_v2_mar02)}),
        ])
        print("\n=== Explainability on dims 29–31 (v2) ===")
        explain = explainability_check(model_v2, dec_v2, scaler_v2)
    else:
        print("\n[v2] checkpoint not found — scaler regression only. Train with --tag v2.")

    print("\n=== BENCHMARK (quote per-row; do not pool 21 Feb with 01 Mar) ===")
    hdr = f"{'slice':42s} {'ver':12s} {'n':>6s} {'MSE':>10s} {'ΔMSE':>8s} {'MITRE_acc':>10s} {'mF1':>8s} {'C2_R':>8s}"
    print(hdr)
    for r in rows:
        mse = "n/a" if r.get("lstm_mse") is None else f"{r['lstm_mse']:10.4f}"
        dms = "n/a" if r.get("mse_reduction_vs_persist") is None else f"{r['mse_reduction_vs_persist']:8.3f}"
        acc = "n/a" if r.get("mitre_accuracy") is None else f"{r['mitre_accuracy']:10.3f}"
        mf1 = "n/a" if r.get("mitre_macro_f1") is None else f"{r['mitre_macro_f1']:8.3f}"
        c2 = r.get("c2") or {}
        c2s = "n/a" if not c2 else f"{c2.get('recall', 0):8.3f}"
        print(f"{r['slice']:42s} {r['version']:12s} {r['n']:6d} {mse:>10s} {dms:>8s} {acc:>10s} {mf1:>8s} {c2s:>8s}")

    report = {
        "scaler_regression": {k: v for k, v in scal_reg.items() if k != "per_dim"},
        "scaler_regression_per_dim": scal_reg.get("per_dim"),
        "rows": rows,
        "explainability": explain,
        "note": (
            "v1 artifacts were not overwritten. Combined scaler was fit on real-train + "
            "synthetic-train only. 02 Mar is the one real held-out C2 day. "
            "Synthetic fixtures stand in until the 20-persona generator lands."
        ),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
