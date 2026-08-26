"""Turn CIC-IDS CSV(s) into (S_t, S_{t+1}) windows for world-model training."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.world_model.cic_schema import CIC2018_COLUMNS, INPUT_DIM, STATE_FEATURE_ORDER
from src.world_model.dataset import concat_window_bundles
from src.world_model.windows import WindowConfig, build_state_windows, load_cic_csv


def generate_demo_flows(
    n_windows: int = 24,
    flows_per_window: int = 8,
    t0: datetime | None = None,
) -> pd.DataFrame:
    """
    Schema-faithful synthetic CIC-IDS2018 rows for offline tests / demo.
    Timeline: benign → SSH brute (T1110) → sequential port scan / infiltration.
    """
    rng = np.random.default_rng(42)
    t0 = t0 or datetime(2018, 2, 28, 10, 0, 0, tzinfo=timezone.utc)
    rows = []
    for w in range(n_windows):
        ts = t0 + timedelta(seconds=5 * w)
        if w < 8:
            label, ports, syn = "Benign", [80, 443, 53], 0
        elif w < 14:
            label, ports, syn = "SSH-Bruteforce", [22], 1
        elif w < 16:
            label, ports, syn = "Infilteration", list(range(20 + w, 20 + w + 12)), 1
        else:
            label, ports, syn = "Benign", [80, 443, 53], 0
        for i in range(flows_per_window):
            port = int(ports[i % len(ports)])
            row = {c: 0 for c in CIC2018_COLUMNS}
            row["Dst Port"] = port
            row["Protocol"] = 6
            row["Timestamp"] = ts.strftime("%d/%m/%Y %H:%M:%S")
            row["Flow Duration"] = float(rng.integers(1000, 80_000))
            row["Tot Fwd Pkts"] = int(rng.integers(1, 12))
            row["Tot Bwd Pkts"] = int(rng.integers(0, 10))
            row["TotLen Fwd Pkts"] = float(rng.integers(40, 2000))
            row["TotLen Bwd Pkts"] = float(rng.integers(0, 4000))
            row["Flow IAT Mean"] = float(rng.integers(100, 5000))
            row["Flow IAT Std"] = float(rng.integers(10, 800))
            row["Flow IAT Max"] = float(rng.integers(1000, 20_000))
            row["SYN Flag Cnt"] = syn
            row["ACK Flag Cnt"] = 1
            row["FIN Flag Cnt"] = 0
            row["RST Flag Cnt"] = 0
            row["PSH Flag Cnt"] = int(label != "Benign")
            row["URG Flag Cnt"] = 0
            row["Down/Up Ratio"] = 1.0
            row["Pkt Len Mean"] = float(rng.integers(40, 600))
            row["Pkt Len Std"] = float(rng.integers(5, 120))
            row["Pkt Len Max"] = float(rng.integers(200, 1460))
            row["Init Fwd Win Byts"] = 8192
            row["Init Bwd Win Byts"] = 8192
            row["Fwd PSH Flags"] = int(label != "Benign")
            row["Label"] = label
            rows.append(row)
    return pd.DataFrame(rows)


def discover_cic_csvs(csv_dir: Path) -> list[Path]:
    files = sorted(
        p for p in csv_dir.rglob("*.csv")
        if p.is_file() and not p.name.startswith(".")
    )
    return files


def _fmt_ts(unix: float) -> str:
    return datetime.fromtimestamp(float(unix), tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def process_csv_file(
    path: Path,
    cfg: WindowConfig,
    nrows: int | None = None,
) -> dict:
    """Load one CIC CSV, window it, then drop the raw DataFrame."""
    print(f"[load] {path}")
    df = load_cic_csv(str(path), nrows=nrows)
    n_rows = len(df)
    n_cols = len(df.columns)
    print(f"       rows={n_rows}  cols={n_cols}")
    bundle = build_state_windows(
        df,
        cfg,
        source_file=path.name,
        require_real_timestamps=True,
    )
    del df
    n_pairs = int(bundle["states"].shape[0])
    days = sorted(set(map(str, bundle["day_id"])))
    print(
        f"[day]  {', '.join(days)}  file={path.name}\n"
        f"       ts={_fmt_ts(bundle['timestamps'].min())} → {_fmt_ts(bundle['timestamps'].max())}\n"
        f"       pairs={n_pairs}  dim={bundle['states'].shape[1]}  "
        f"attack_now={float(bundle['attack_now'].mean()):.3f}  "
        f"attack_within_k={float(bundle['attack_within_k'].mean()):.3f}"
    )
    bundle["_n_raw_rows"] = np.array([n_rows])
    return bundle


def save_windows(bundle: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {k: v for k, v in bundle.items() if not str(k).startswith("_")}
    np.savez_compressed(out_path, **payload)
    unique_days = sorted(set(map(str, bundle["day_id"]))) if "day_id" in bundle else []
    meta = {
        "n_pairs": int(bundle["states"].shape[0]),
        "input_dim": int(bundle["states"].shape[1]),
        "feature_names": STATE_FEATURE_ORDER,
        "window_seconds": int(bundle["window_seconds"][0]),
        "horizon_k": int(bundle["horizon_k"][0]),
        "days": unique_days,
        "attack_rate_now": float(bundle["attack_now"].mean()),
        "attack_within_k_rate": float(bundle["attack_within_k"].mean()),
        "infiltration_within_k_rate": float(bundle["infiltration_within_k"].mean()),
        "pre_attack_rate": float(bundle["pre_attack"].mean()),
    }
    meta_path = out_path.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[saved] {out_path}  pairs={meta['n_pairs']}  dim={INPUT_DIM}  days={unique_days}")
    print(f"[meta]  {meta_path}")
    print(f"        attack_now={meta['attack_rate_now']:.3f}  "
          f"attack_within_k={meta['attack_within_k_rate']:.3f}  "
          f"infiltration_within_k={meta['infiltration_within_k_rate']:.3f}  "
          f"pre_attack={meta['pre_attack_rate']:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build S_t windows from CIC-IDS CSV")
    parser.add_argument(
        "--csv",
        action="append",
        default=[],
        help="Path to a CIC-IDS2017/2018 flow CSV (repeatable)",
    )
    parser.add_argument(
        "--csv-dir",
        help="Directory of CIC CSVs (recursive). Each file is windowed separately.",
    )
    parser.add_argument("--demo", action="store_true", help="Use synthetic CIC-schema flows")
    parser.add_argument("--out", default="data/processed/state_windows.npz")
    parser.add_argument("--window-seconds", type=int, default=5)
    parser.add_argument("--horizon-k", type=int, default=6)
    parser.add_argument("--nrows", type=int, default=None, help="Optional row cap per file")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    out = Path(args.out)
    if not out.is_absolute():
        out = root / out

    cfg = WindowConfig(window_seconds=args.window_seconds, horizon_k=args.horizon_k)
    paths: list[Path] = []
    for raw in args.csv:
        p = Path(raw)
        if not p.is_absolute():
            p = root / p
        paths.append(p)
    if args.csv_dir:
        d = Path(args.csv_dir)
        if not d.is_absolute():
            d = root / d
        if not d.is_dir():
            raise SystemExit(f"--csv-dir is not a directory: {d}")
        found = discover_cic_csvs(d)
        print(f"[discover] {d} → {len(found)} csv file(s)")
        for p in found:
            print(f"           {p}")
        paths.extend(found)

    # De-duplicate while preserving order
    seen: set[Path] = set()
    unique_paths: list[Path] = []
    for p in paths:
        key = p.resolve()
        if key in seen:
            continue
        seen.add(key)
        unique_paths.append(p)

    if args.demo:
        df = generate_demo_flows()
        print(f"[demo] generated {len(df)} synthetic CIC-IDS2018 rows")
        bundle = build_state_windows(df, cfg, source_file="demo.csv")
        save_windows(bundle, out)
        return

    if not unique_paths:
        raise SystemExit("Pass --csv PATH, --csv-dir DIR, or --demo")

    bundles = []
    total_rows = 0
    for path in unique_paths:
        if not path.is_file():
            raise SystemExit(f"CSV not found: {path}")
        bundle = process_csv_file(path, cfg, nrows=args.nrows)
        total_rows += int(bundle.pop("_n_raw_rows")[0])
        bundles.append(bundle)

    combined = concat_window_bundles(bundles)
    print(
        f"[total] files={len(bundles)}  raw_rows={total_rows}  "
        f"pairs={len(combined['states'])}  days={sorted(set(map(str, combined['day_id'])))}"
    )
    save_windows(combined, out)


if __name__ == "__main__":
    main()
