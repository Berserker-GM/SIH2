"""Build time-windowed network states S_t from CIC-IDS flow CSVs."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import json

import numpy as np
import pandas as pd

from src.world_model.cic_schema import STATE_FEATURE_ORDER, bind_columns
from src.world_model.labels import map_label, pick_window_stage


@dataclass(frozen=True)
class WindowConfig:
    window_seconds: int = 5
    stride_seconds: int = 5
    horizon_k: int = 6  # lookahead windows (30s at default Δt)


def _col(df: pd.DataFrame, bound: dict[str, str], field: str) -> pd.Series:
    if field not in bound:
        return pd.Series(0, index=df.index, dtype="float64")
    return pd.to_numeric(df[bound[field]], errors="coerce")


def _shannon_entropy(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    counts = np.array(list(Counter(values.tolist()).values()), dtype=np.float64)
    probs = counts / counts.sum()
    return float(-(probs * np.log2(np.clip(probs, 1e-12, 1.0))).sum())


def _port_scan_score(ports: np.ndarray) -> float:
    """Sequential scan → high; randomised many-port access also high."""
    unique = np.unique(ports)
    if unique.size < 8:
        return 0.0
    diffs = np.diff(np.sort(unique.astype(np.int64)))
    sequential = float((diffs == 1).mean()) if diffs.size else 0.0
    spread = min(1.0, unique.size / 40.0)
    return float(max(sequential, spread * min(1.0, _shannon_entropy(unique) / 5.0)))


def parse_timestamps(series: pd.Series) -> pd.Series:
    formats = [
        "%d/%m/%Y %H:%M:%S",
        "%d/%m/%Y %H:%M",
        "%m/%d/%Y %H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%d/%m/%Y %I:%M:%S %p",
    ]
    parsed = pd.to_datetime(series, errors="coerce", utc=True, dayfirst=True)
    if parsed.notna().mean() >= 0.9:
        return parsed
    for fmt in formats:
        parsed = pd.to_datetime(series, format=fmt, errors="coerce", utc=True)
        if parsed.notna().mean() >= 0.5:
            return parsed
    return pd.to_datetime(series, errors="coerce", utc=True)


LIVE_FULL_LOAD_BYTES = 80 * 1024 * 1024
LIVE_SLICE_SECONDS = 90.0
_CIC_NA = ["Infinity", "infinity", "NaN", "nan", "inf", "-inf", ""]


def _live_slice_mode(mode: str | None) -> str:
    key = str(mode or "latest").strip().casefold()
    if key in {"attack", "ddos", "peak", "densest"}:
        return "attack"
    return "latest"


def _finalize_cic_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    bound = bind_columns(list(df.columns))
    if "label" in bound:
        label_col = bound["label"]
        df = df[df[label_col].astype(str).str.strip().str.casefold() != "label"]
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df.reset_index(drop=True)


def load_cic_csv(path: str, nrows: Optional[int] = None) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        low_memory=False,
        nrows=nrows,
        na_values=_CIC_NA,
    )
    return _finalize_cic_frame(df)


def _sniff_ts_label_columns(path: str | Path) -> tuple[str, str]:
    header = pd.read_csv(path, nrows=0)
    header.columns = [str(c).strip() for c in header.columns]
    bound = bind_columns(list(header.columns))
    if "timestamp" not in bound or "label" not in bound:
        raise ValueError("CSV needs Timestamp and Label columns for live inference")
    return bound["timestamp"], bound["label"]


def _pick_live_interval(
    ts: pd.Series,
    labels: pd.Series,
    slice_seconds: float,
    window_seconds: int = 5,
    mode: str = "latest",
) -> tuple[pd.Timestamp, pd.Timestamp, str]:
    """Choose a ~90s bin for huge CIC days.

    Default is the latest timestamps (world-model “now”). Attack-seeking
    (densest labeled-attack interval) is opt-in — using it as the default
    made every CIC day look like a threat.
    """
    valid = ts.notna()
    ts = ts[valid]
    labels = labels[valid]
    if ts.empty:
        raise ValueError("No parseable timestamps")
    lab = labels.astype(str).str.strip().str.casefold()
    attack = lab.ne("benign") & lab.ne("label") & lab.ne("")
    tmin = ts.min()
    n_bins = max(8, int(round(slice_seconds / window_seconds)))
    wid = ((ts - tmin).dt.total_seconds() // window_seconds).astype(np.int64)
    w0 = int(wid.min())
    w1 = int(wid.max())
    span = np.arange(w0, w1 + 1)
    atk = (
        pd.Series(attack.to_numpy(dtype=np.int64), index=wid.to_numpy())
        .groupby(level=0)
        .sum()
        .reindex(span, fill_value=0)
        .to_numpy()
    )
    kernel = np.ones(n_bins, dtype=np.float64)
    if atk.size < n_bins:
        t0 = tmin
        t1 = tmin + pd.Timedelta(seconds=slice_seconds)
        return t0, t1, f"short capture; using first {slice_seconds:.0f}s"
    flows = (
        pd.Series(1, index=wid.to_numpy())
        .groupby(level=0)
        .sum()
        .reindex(span, fill_value=0)
        .to_numpy()
    )
    flow_roll = np.convolve(flows, kernel, mode="valid")
    atk_roll = np.convolve(atk, kernel, mode="valid")
    mode = _live_slice_mode(mode)
    if mode == "attack":
        start_i = int(np.argmax(atk_roll))
        if float(atk_roll[start_i]) <= 0:
            start_i = int(np.argmax(flow_roll))
            reason = f"no attack labels; densest {slice_seconds:.0f}s by flow count"
        else:
            reason = (
                f"densest {slice_seconds:.0f}s attack window "
                f"({int(atk_roll[start_i])} attack flows)"
            )
    else:
        start_i = int(len(flow_roll) - 1)
        min_flows = 8.0
        while start_i > 0 and float(flow_roll[start_i]) < min_flows:
            start_i -= 1
        reason = (
            f"latest {slice_seconds:.0f}s of capture "
            f"({int(flow_roll[start_i])} flows, {int(atk_roll[start_i])} labeled attack)"
        )
    start_wid = int(span[start_i])
    t0 = tmin + pd.Timedelta(seconds=float(start_wid * window_seconds))
    t1 = t0 + pd.Timedelta(seconds=slice_seconds)
    return t0, t1, reason


def _read_timestamp_label(path: str | Path) -> tuple[pd.Series, pd.Series]:
    ts_col, lab_col = _sniff_ts_label_columns(path)
    parts_ts = []
    parts_lab = []
    for chunk in pd.read_csv(
        path,
        usecols=[ts_col, lab_col],
        chunksize=250_000,
        low_memory=False,
        na_values=_CIC_NA,
    ):
        chunk.columns = [str(c).strip() for c in chunk.columns]
        parts_ts.append(parse_timestamps(chunk[ts_col]))
        parts_lab.append(chunk[lab_col].astype(str))
    if not parts_ts:
        raise ValueError("CSV has no rows")
    return pd.concat(parts_ts, ignore_index=True), pd.concat(parts_lab, ignore_index=True)


def _read_time_slice(path: str | Path, t0: pd.Timestamp, t1: pd.Timestamp) -> pd.DataFrame:
    ts_col, _lab_col = _sniff_ts_label_columns(path)
    kept = []
    for chunk in pd.read_csv(
        path,
        chunksize=80_000,
        low_memory=False,
        na_values=_CIC_NA,
    ):
        chunk.columns = [str(c).strip() for c in chunk.columns]
        ts = parse_timestamps(chunk[ts_col])
        mask = ts.notna() & (ts >= t0) & (ts < t1)
        if bool(mask.any()):
            kept.append(chunk.loc[mask])
    if not kept:
        raise ValueError(
            f"No flows in live slice {t0.isoformat()} → {t1.isoformat()}"
        )
    return _finalize_cic_frame(pd.concat(kept, ignore_index=True))


def _slice_cache_paths(src: Path, slice_seconds: float, mode: str = "latest") -> tuple[Path, Path]:
    from src.world_model.train import ROOT
    folder = ROOT / "data" / "processed" / "live_slices"
    folder.mkdir(parents=True, exist_ok=True)
    stem = f"{src.stem}_s{int(slice_seconds)}_{_live_slice_mode(mode)}"
    return folder / f"{stem}.csv", folder / f"{stem}.json"


def load_cic_csv_live(
    path: str | Path,
    *,
    slice_seconds: float = LIVE_SLICE_SECONDS,
    full_load_bytes: int = LIVE_FULL_LOAD_BYTES,
    slice_mode: str = "latest",
) -> tuple[pd.DataFrame, dict]:
    """
    Load a CIC CSV for LSTM inference.

    Small files are read whole. Multi-GB unsorted days are reduced to a
    contiguous ~90s window. Default is the latest timestamps (S_t = now),
    not the densest attack interval.
    """
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"CSV not found: {path}")
    file_bytes = int(path.stat().st_size)
    mode = _live_slice_mode(slice_mode)
    meta: dict = {
        "file_bytes": file_bytes,
        "truncated": False,
        "slice_seconds": float(slice_seconds),
        "slice_mode": mode,
        "note": "full file",
    }
    if file_bytes <= full_load_bytes:
        df = load_cic_csv(str(path))
        meta["n_rows"] = int(len(df))
        return df, meta

    cache_csv, cache_json = _slice_cache_paths(path, slice_seconds, mode)
    if cache_csv.is_file() and cache_csv.stat().st_mtime >= path.stat().st_mtime:
        df = load_cic_csv(str(cache_csv))
        if cache_json.is_file():
            saved = json.loads(cache_json.read_text(encoding="utf-8"))
            saved["n_rows"] = int(len(df))
            saved["cached"] = True
            saved["slice_mode"] = saved.get("slice_mode") or mode
            return df, saved
        meta.update({
            "truncated": True,
            "n_rows": int(len(df)),
            "cached": True,
            "note": f"cached 90s live slice ({mode})",
        })
        return df, meta

    ts, labels = _read_timestamp_label(path)
    t0, t1, reason = _pick_live_interval(ts, labels, slice_seconds, mode=mode)
    df = _read_time_slice(path, t0, t1)
    meta.update({
        "truncated": True,
        "n_rows": int(len(df)),
        "t0": t0.isoformat(),
        "t1": t1.isoformat(),
        "note": reason,
        "cached": False,
        "slice_mode": mode,
    })
    df.to_csv(cache_csv, index=False)
    cache_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return df, meta


def _aggregate_window(g: pd.DataFrame, bound: dict[str, str]) -> np.ndarray:
    ports = _col(g, bound, "dst_port").fillna(0).to_numpy()
    proto = _col(g, bound, "protocol").fillna(0).to_numpy()
    bytes_fwd = float(_col(g, bound, "totlen_fwd").fillna(0).sum())
    bytes_bwd = float(_col(g, bound, "totlen_bwd").fillna(0).sum())
    total_bytes = bytes_fwd + bytes_bwd
    vec = np.zeros(len(STATE_FEATURE_ORDER), dtype=np.float32)
    vec[0] = float(len(g))
    vec[1] = float(np.unique(ports).size)
    vec[2] = _shannon_entropy(ports)
    vec[3] = float((proto == 6).mean()) if proto.size else 0.0
    vec[4] = float((proto == 17).mean()) if proto.size else 0.0
    vec[5] = bytes_fwd
    vec[6] = bytes_bwd
    vec[7] = bytes_bwd / total_bytes if total_bytes > 0 else 0.0
    vec[8] = float(_col(g, bound, "tot_fwd_pkts").fillna(0).sum())
    vec[9] = float(_col(g, bound, "tot_bwd_pkts").fillna(0).sum())
    dur = _col(g, bound, "flow_duration").fillna(0)
    vec[10] = float(dur.mean())
    vec[11] = float(dur.max())
    iat_mean = _col(g, bound, "flow_iat_mean").fillna(0)
    iat_std = _col(g, bound, "flow_iat_std").fillna(0)
    iat_max = _col(g, bound, "flow_iat_max").fillna(0)
    vec[12] = float(iat_mean.mean())
    vec[13] = float(iat_std.mean())
    vec[14] = float(iat_max.max())
    vec[15] = float(_col(g, bound, "syn_flag").fillna(0).sum())
    vec[16] = float(_col(g, bound, "ack_flag").fillna(0).sum())
    vec[17] = float(_col(g, bound, "fin_flag").fillna(0).sum())
    vec[18] = float(_col(g, bound, "rst_flag").fillna(0).sum())
    vec[19] = float(_col(g, bound, "psh_flag").fillna(0).sum())
    vec[20] = float(_col(g, bound, "urg_flag").fillna(0).sum())
    vec[21] = float(_col(g, bound, "down_up_ratio").fillna(0).mean())
    vec[22] = float(_col(g, bound, "pkt_len_mean").fillna(0).mean())
    vec[23] = float(_col(g, bound, "pkt_len_std").fillna(0).mean())
    vec[24] = float(_col(g, bound, "pkt_len_max").fillna(0).max())
    vec[25] = float(_col(g, bound, "init_fwd_win").fillna(0).mean())
    vec[26] = float(_col(g, bound, "init_bwd_win").fillna(0).mean())
    vec[27] = float(_col(g, bound, "fwd_psh").fillna(0).sum())
    vec[28] = _port_scan_score(ports)
    vec[29] = 0.0  # ttl_variance — PCAP
    vec[30] = 0.0  # ip_fragment_flags — PCAP
    vec[31] = 0.0  # retransmit_count — PCAP
    return vec


def build_state_windows(
    df: pd.DataFrame,
    config: WindowConfig | None = None,
    *,
    source_file: str = "",
    require_real_timestamps: bool = True,
    include_graphs: bool = False,
) -> dict:
    """
    Convert labelled CIC flows into (S_t, S_{t+1}) pairs.

    Call once per CSV / capture day. Do not concatenate raw multi-day
    DataFrames before this function — window_id is elapsed time within
    *this* frame only.

    S_t is always the frozen 32-d vector. When include_graphs=True, also
    emit G_t (communication graph) for the same bins — used by live
    inference, not by LSTM training NPZ.

    Returns a dict of numpy arrays ready for world-model training.
    """
    cfg = config or WindowConfig()
    if df.empty:
        raise ValueError("No rows to window")

    bound = bind_columns(list(df.columns))
    if "label" not in bound:
        raise ValueError("CSV has no Label column")

    work = df.copy()
    if "timestamp" not in bound:
        raise ValueError(
            "CSV has no Timestamp column; refusing fake row-index timestamps."
        )
    ts = parse_timestamps(work[bound["timestamp"]])
    parse_frac = float(ts.notna().mean()) if len(ts) else 0.0
    if require_real_timestamps and parse_frac < 0.9:
        raise ValueError(
            f"Unparseable timestamps ({parse_frac:.1%} valid). "
            "Refusing fake row-index clock for world-model windows."
        )
    if ts.notna().sum() == 0:
        raise ValueError("No parseable timestamps")

    work["_ts"] = ts
    work = work.dropna(subset=["_ts"]).sort_values("_ts")
    if work.empty:
        raise ValueError("No parseable timestamps")

    t0 = work["_ts"].iloc[0]
    elapsed = (work["_ts"] - t0).dt.total_seconds().to_numpy()
    work["_wid"] = (elapsed // cfg.window_seconds).astype(np.int64)

    grouped = []
    graph_fn = None
    if include_graphs:
        from src.world_model.state_graph import build_window_graph as graph_fn
    for wid, g in work.groupby("_wid", sort=True):
        mapped = [map_label(x) for x in g[bound["label"]].tolist()]
        stage = pick_window_stage(mapped)
        ts0 = pd.Timestamp(g["_ts"].iloc[0])
        if ts0.tzinfo is None:
            ts0 = ts0.tz_localize("UTC")
        row = {
            "window_id": int(wid),
            "timestamp": ts0.timestamp(),
            "day_id": ts0.strftime("%Y-%m-%d"),
            "source_file": source_file,
            "state": _aggregate_window(g, bound),
            "attack_now": int(any(m["is_attack"] for m in mapped)),
            "infiltration_now": int(any(m["is_infiltration"] for m in mapped)),
            "stage_id": stage["stage_id"],
            "technique_id": stage["technique_id"],
            "raw_label": stage["raw_label"],
        }
        if include_graphs:
            row["graph"] = graph_fn(g, bound)
        grouped.append(row)

    if len(grouped) < 2:
        raise ValueError("Need at least 2 time windows")

    by_id = {row["window_id"]: row for row in grouped}
    ids = [row["window_id"] for row in grouped]

    states, next_states = [], []
    attack_now, attack_k, infil_k = [], [], []
    stage_ids, techniques, timestamps, window_ids = [], [], [], []
    day_ids, source_files, graphs = [], [], []

    for i, wid in enumerate(ids[:-1]):
        nxt = ids[i + 1]
        # Skip non-adjacent windows when the hole is larger than 15s (3 bins).
        if nxt != wid + 1 and (nxt - wid) * cfg.window_seconds > cfg.window_seconds * 3:
            continue
        future_ids = [fid for fid in ids if wid < fid <= wid + cfg.horizon_k]
        states.append(by_id[wid]["state"])
        next_states.append(by_id[nxt]["state"])
        attack_now.append(by_id[wid]["attack_now"])
        attack_k.append(int(any(by_id[fid]["attack_now"] for fid in future_ids)))
        infil_k.append(int(any(by_id[fid]["infiltration_now"] for fid in future_ids)))
        stage_ids.append(by_id[wid]["stage_id"])
        techniques.append(by_id[wid]["technique_id"])
        timestamps.append(by_id[wid]["timestamp"])
        window_ids.append(wid)
        day_ids.append(by_id[wid]["day_id"])
        source_files.append(by_id[wid]["source_file"])
        if include_graphs:
            graphs.append(by_id[wid]["graph"])

    if not states:
        raise ValueError("No consecutive window pairs produced")

    attack_now_arr = np.array(attack_now, dtype=np.int8)
    attack_k_arr = np.array(attack_k, dtype=np.int8)
    infil_k_arr = np.array(infil_k, dtype=np.int8)

    out = {
        "states": np.stack(states),
        "next_states": np.stack(next_states),
        "attack_now": attack_now_arr,
        "attack_within_k": attack_k_arr,
        "infiltration_within_k": infil_k_arr,
        "pre_attack": ((attack_k_arr == 1) & (attack_now_arr == 0)).astype(np.int8),
        "stage_id": np.array(stage_ids, dtype=np.int8),
        "technique_id": np.array(techniques, dtype=object),
        "timestamps": np.array(timestamps, dtype=np.float64),
        "window_ids": np.array(window_ids, dtype=np.int64),
        "day_id": np.array(day_ids, dtype="U10"),
        "source_file": np.array(source_files, dtype=object),
        "feature_names": np.array(STATE_FEATURE_ORDER),
        "window_seconds": np.array([cfg.window_seconds]),
        "horizon_k": np.array([cfg.horizon_k]),
    }
    if include_graphs:
        out["graphs"] = np.array(graphs, dtype=object)
    return out
