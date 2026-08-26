"""Frozen world-model runtime for the dashboard API.

Does not train. Does not require Redis.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch.nn as nn

from src.world_model.cic_schema import STATE_FEATURE_ORDER, PACKET_LEVEL_SPEC
from src.world_model.dataset import (
    DEFAULT_TEST_DAYS,
    DEFAULT_TRAIN_DAYS,
    DEFAULT_VAL_DAYS,
    Scaler,
    day_split,
    load_npz,
)
from src.world_model.explain import LSTM_ATTACK_THRESHOLD, explain_forecast
from src.world_model.labels import STAGE_NAMES, technique_for_stage
from src.world_model.mitre_decode import StageDecoder, fit_stage_decoder
from src.world_model.train import ROOT
from src.world_model.windows import WindowConfig, build_state_windows, load_cic_csv_live

DEFAULT_EXAMPLES = ROOT / "src" / "world_model" / "models" / "explain_examples.json"
DEFAULT_CKPT = ROOT / "src" / "world_model" / "models" / "world_lstm.pt"
DEFAULT_SCALER = ROOT / "src" / "world_model" / "models" / "scaler.npz"
DEFAULT_NPZ = ROOT / "data" / "processed" / "state_windows_multiday.npz"
MAX_GAP_SECONDS = 15.0
MAX_UPLOAD_BYTES = 80 * 1024 * 1024

LOCAL_CIC_ALIASES: dict[str, tuple[str, ...]] = {
    "ddos-loic-http": (
        "CIC-IDS2018/Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv",
        "CIC-IDS2018/Tuesday-20-02-2018_TrafficForML_CICFlowMeter.csv",
        "Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv",
        "Tuesday-20-02-2018_TrafficForML_CICFlowMeter.csv",
    ),
}


def resolve_local_csv(name: str) -> Path:
    """Resolve a CIC CSV already on disk under data/raw (no browser upload)."""
    raw = (ROOT / "data" / "raw").resolve()
    key = str(name).strip().replace("\\", "/")
    candidates: list[Path] = []
    for rel in LOCAL_CIC_ALIASES.get(key.casefold(), ()):
        candidates.append(raw / rel)
    rel = Path(key)
    candidates.append((raw / rel).resolve())
    candidates.append((raw / rel.name).resolve())
    seen: set[Path] = set()
    for cand in candidates:
        try:
            rp = cand.resolve()
        except OSError:
            continue
        if rp in seen:
            continue
        seen.add(rp)
        if rp.is_file() and (rp.parent == raw or raw in rp.parents):
            return rp
    raise ValueError(
        f"No local CIC CSV for {name!r}. Expected under data/raw/. "
        "20 Feb DDoS file is Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv"
    )


def resolve_local_pcap(name: str) -> Path:
    """Resolve a PCAP already on disk under data/raw."""
    raw = (ROOT / "data" / "raw").resolve()
    key = str(name).strip().replace("\\", "/")
    rel = Path(key)
    candidates = [(raw / rel).resolve(), (raw / rel.name).resolve()]
    seen: set[Path] = set()
    for cand in candidates:
        try:
            rp = cand.resolve()
        except OSError:
            continue
        if rp in seen:
            continue
        seen.add(rp)
        if rp.is_file() and (rp.parent == raw or raw in rp.parents):
            if rp.suffix.lower() not in {".pcap", ".pcapng", ".cap"}:
                raise ValueError(f"Not a PCAP: {rp.name}")
            return rp
    raise ValueError(f"No local PCAP for {name!r} under data/raw/")


LOCAL_CIC_FOLDERS = ("CIC-IDS2018", "CIC2018", "CIC-IDS-2018")
LOCAL_CIC_DIR_REL = "data/raw/CIC-IDS2018"


def local_cic_dir() -> Path:
    """Canonical drop folder for on-disk CIC CSVs (created if missing)."""
    d = ROOT / "data" / "raw" / "CIC-IDS2018"
    d.mkdir(parents=True, exist_ok=True)
    return d


def list_local_csv() -> list[dict[str, Any]]:
    """CSVs under data/raw. CIC-IDS2018 / CIC2018 files are listed first."""
    raw = (ROOT / "data" / "raw").resolve()
    local_cic_dir()
    if not raw.is_dir():
        return []
    seen: set[Path] = set()
    out: list[dict[str, Any]] = []
    for p in raw.rglob("*.csv"):
        if not p.is_file():
            continue
        try:
            rp = p.resolve()
        except OSError:
            continue
        if rp in seen:
            continue
        if rp.parent != raw and raw not in rp.parents:
            continue
        seen.add(rp)
        rel = str(rp.relative_to(raw)).replace("\\", "/")
        top = rel.split("/", 1)[0]
        out.append({
            "name": rp.name,
            "rel": rel,
            "bytes": int(rp.stat().st_size),
            "in_cic_folder": top in LOCAL_CIC_FOLDERS,
        })

    def sort_key(row: dict[str, Any]) -> tuple:
        return (0 if row["in_cic_folder"] else 1, str(row["rel"]).lower())

    out.sort(key=sort_key)
    return out


def jsonable(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    return obj


def load_examples(path: Path | None = None) -> list[dict]:
    p = path or DEFAULT_EXAMPLES
    if not p.is_file():
        return []
    raw = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        return list(raw.get("examples") or [])
    return list(raw)


def last_consecutive_history(
    states: np.ndarray,
    timestamps: np.ndarray,
    seq_len: int = 8,
    max_gap_seconds: float = MAX_GAP_SECONDS,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Last length-`seq_len` windows with strictly increasing ≤15s gaps.

    Returns (history, timestamps, end_index) where end_index is exclusive,
    so S_t is row end_index-1.
    """
    n = len(states)
    if n < seq_len:
        raise ValueError(f"Need at least {seq_len} state windows, got {n}")
    ts = np.asarray(timestamps, dtype=np.float64)
    for end in range(n, seq_len - 1, -1):
        sl = slice(end - seq_len, end)
        win_ts = ts[sl]
        dts = np.diff(win_ts)
        if np.any(dts <= 0) or np.any(dts > max_gap_seconds + 1e-6):
            continue
        return np.asarray(states[sl], dtype=np.float32), win_ts, int(end)
    raise ValueError(
        f"No {seq_len} consecutive windows with adjacent gaps ≤ {max_gap_seconds}s"
    )


def load_frozen_runtime() -> "WorldModelRuntime":
    """Load frozen LSTM + train-only scaler/decoder. Does not retrain."""
    from src.world_model.eval_kstep import _load_model

    examples = load_examples()
    if not DEFAULT_CKPT.is_file() or not DEFAULT_SCALER.is_file():
        return WorldModelRuntime(examples=examples)
    model, _ckpt = _load_model(DEFAULT_CKPT)
    scaler = Scaler.load(DEFAULT_SCALER)
    decoder = None
    if DEFAULT_NPZ.is_file():
        data = load_npz(DEFAULT_NPZ)
        train_i, _, _ = day_split(
            data["day_id"], DEFAULT_TRAIN_DAYS, DEFAULT_VAL_DAYS, DEFAULT_TEST_DAYS,
        )
        decoder = fit_stage_decoder(
            scaler.transform(data["states"][train_i]).astype(np.float32),
            data["stage_id"][train_i],
        )
    return WorldModelRuntime(
        model=model, scaler=scaler, decoder=decoder, examples=examples,
    )


@dataclass
class WorldModelRuntime:
    model: nn.Module | None = None
    scaler: Scaler | None = None
    decoder: StageDecoder | None = None
    examples: list[dict] = field(default_factory=list)
    attack_threshold: float = LSTM_ATTACK_THRESHOLD
    input_dim: int = 32
    seq_len: int = 8
    k: int = 6

    @property
    def frozen(self) -> bool:
        return True

    def status(self) -> dict:
        return {
            "frozen": True,
            "ready_for_live": self.model is not None and self.decoder is not None and self.scaler is not None,
            "n_examples": len(self.examples),
            "input_dim": self.input_dim,
            "seq_len": self.seq_len,
            "k": self.k,
            "horizon_seconds": self.k * 5,
            "attack_threshold": self.attack_threshold,
            "feature_names": list(STATE_FEATURE_ORDER),
            "packet_level_spec": [dict(row) for row in PACKET_LEVEL_SPEC],
            "note": (
                "LSTM weights are not updated. POST /upload accepts CIC CSVs up to 80 MB "
                "plus an optional PCAP sidecar (TTL / fragment / retransmit; not injected "
                "into frozen S_t). Multi-GB days (20 Feb LOIC-HTTP) stay on disk — POST /local. "
                "Live inference slices a contiguous ~90s window, builds 32-d S_t plus G_t, "
                "then runs the frozen LSTM."
            ),
            "max_upload_mb": 80,
            "local_csv_dir": LOCAL_CIC_DIR_REL,
            "local_csv": list_local_csv(),
        }

    def example(self, i: int = 0) -> dict:
        if not self.examples:
            raise ValueError("No world-model examples loaded")
        item = self.examples[int(i) % len(self.examples)]
        out = jsonable(item)
        out["index"] = int(i) % len(self.examples)
        out["n_examples"] = len(self.examples)
        out["source"] = out.get("source") or "replay"
        return out

    def forecast(self, history: np.ndarray) -> dict:
        if self.model is None or self.decoder is None:
            raise RuntimeError("Live LSTM is not loaded")
        hist = np.asarray(history, dtype=np.float32)
        if hist.shape != (self.seq_len, self.input_dim):
            raise ValueError(
                f"history must be ({self.seq_len}, {self.input_dim}), got {hist.shape}"
            )
        bundle = explain_forecast(
            self.model,
            self.decoder,
            hist,
            k=self.k,
            attack_threshold=self.attack_threshold,
        )
        return jsonable(bundle)

    def infer_from_csv(
        self,
        csv_path: str | Path,
        source_file: str = "",
        pcap_path: str | Path | None = None,
        slice_mode: str = "latest",
    ) -> dict:
        """CSV → (90s slice if huge) → windows → frozen scaler → K=6 forecast."""
        if self.model is None or self.decoder is None or self.scaler is None:
            raise RuntimeError("Frozen LSTM / scaler / decoder are not loaded")
        df, live_meta = load_cic_csv_live(csv_path, slice_mode=slice_mode)
        windows = build_state_windows(
            df,
            WindowConfig(window_seconds=5, stride_seconds=5, horizon_k=self.k),
            source_file=source_file or str(csv_path),
            include_graphs=True,
        )
        hist, hist_ts, hist_end = last_consecutive_history(
            windows["states"], windows["timestamps"], seq_len=self.seq_len,
        )
        scaled = self.scaler.transform(hist).astype(np.float32)
        out = self.forecast(scaled)
        names = [str(x) for x in STATE_FEATURE_ORDER]
        s_t = hist[-1]
        window_s = float(np.asarray(windows["window_seconds"]).reshape(-1)[0])
        t_now = float(hist_ts[-1])
        preds = np.asarray(out["forecast_states"], dtype=np.float32)
        probs = np.asarray(out["attack_probs_k"], dtype=np.float64)
        stage_ids = self.decoder.predict(preds)
        timeline = []
        forecast_ts = []
        for i in range(self.k):
            ts = t_now + window_s * (i + 1)
            forecast_ts.append(ts)
            sid = int(stage_ids[i])
            tid, tname = technique_for_stage(sid)
            timeline.append({
                "k": i + 1,
                "seconds_ahead": int(window_s * (i + 1)),
                "timestamp": ts,
                "timestamp_iso": datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                "attack_probability": float(probs[i]),
                "stage": STAGE_NAMES[sid],
                "technique_id": tid,
                "technique_name": tname,
            })
        out["source"] = "upload"
        out["n_pairs"] = int(len(windows["states"]))
        out["n_windows"] = int(len(windows["states"]))
        out["seq_len"] = int(self.seq_len)
        out["input_dim"] = int(self.input_dim)
        out["feature_names"] = names
        out["history_states"] = hist.astype(np.float32)
        out["history_states_scaled"] = scaled
        out["s_t"] = {names[i]: float(s_t[i]) for i in range(len(names))}
        out["history_timestamps"] = [float(x) for x in hist_ts]
        out["t_now"] = t_now
        out["t_now_iso"] = datetime.fromtimestamp(t_now, tz=timezone.utc).isoformat()
        out["forecast_timestamps"] = forecast_ts
        out["timeline"] = timeline
        out["window_seconds"] = window_s
        out["source_file"] = source_file or Path(csv_path).name
        graphs = windows.get("graphs")
        if graphs is not None and len(graphs) >= hist_end:
            out["state_graph"] = graphs[hist_end - 1]
        out["live"] = live_meta
        out["slice_mode"] = live_meta.get("slice_mode") or slice_mode
        st_idx = hist_end - 1
        out["s_t_attack_now"] = int(windows["attack_now"][st_idx])
        out["s_t_cic_stage"] = STAGE_NAMES[int(windows["stage_id"][st_idx])]
        out["packet_level_spec"] = [dict(row) for row in PACKET_LEVEL_SPEC]
        if pcap_path is not None:
            from src.world_model.pcap_features import stats_for_pcap_window

            t1 = t_now + window_s
            pcap_stats = stats_for_pcap_window(pcap_path, t_now, t1)
            pcap_stats["note"] = (
                "PCAP sidecar for S_t only. Frozen LSTM still sees "
                "ttl_variance=ip_fragment_flags=retransmit_count=0."
            )
            out["pcap_stats"] = pcap_stats
        return jsonable(out)
