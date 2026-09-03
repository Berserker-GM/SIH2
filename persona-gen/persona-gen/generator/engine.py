"""
Phase state machine + CSV writer.

Each persona config's `phases` list defines a small state machine (e.g.
probe -> brute -> lateral). We walk it window-by-window, sampling a dwell
time per phase visit, and emit one CSV row every 5 simulated seconds.
`run_id` is fresh per call — callers must never reuse one across a
regenerated persona run (see interface contract).
"""
import csv
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from .schema import CSV_COLUMNS, WINDOW_SECONDS
from .sampler import sample_window, blend_toward_benign

BENIGN_REFERENCE_FLAGS_KEY = "flags"


def _phase_by_name(cfg, name):
    for p in cfg["phases"]:
        if p["name"] == name:
            return p
    raise KeyError(f"unknown phase '{name}' in persona {cfg['persona_id']}")


def _make_benign_reference_row(cfg, rng):
    """A neutral benign-shaped row used only for the overlap/noise blend,
    not emitted as its own row."""
    benign_phase = {
        "name": "_benign_ref", "mitre_stage": "benign", "next": None,
        "flags": {
            "syn_ratio": 0.05, "ack_ratio": 0.6, "fin_ratio": 0.1, "rst_ratio": 0.02,
            "psh_ratio": 0.1, "urg_ratio": 0.01, "tcp_ratio": 0.85,
            "down_up_ratio_mean": 1.0, "down_up_ratio_std": 0.2,
            "port_scan_score_base": 0.0, "port_scan_score_gain": 0.05,
        },
        "timing": {"interarrival_dist": "exponential", "lambda": 1.0, "jitter_pct": 0.2},
        "ports": {"pool": list(range(1024, 65535)), "selection": "randomized"},
        "payload": {"size_dist": "lognormal", "mean": 400, "std": 150},
        "packet_fields": {"ttl_base": 64, "ttl_jitter": 2, "fragment_rate": 0.0, "retransmit_rate": 0.01},
    }
    return sample_window(cfg, benign_phase, rng)


def generate_run(cfg: dict, total_windows: int, run_id: str = None, seed_offset: int = 0):
    """Returns (run_id, list[dict]) where each dict is one full CSV row
    (metadata + 32 features), in schema order."""
    run_id = run_id or f"{cfg['persona_id']}__{uuid.uuid4().hex[:12]}"
    seed = int(cfg["base_seed"]) + seed_offset
    rng = np.random.default_rng(seed)

    phases_by_name = {p["name"]: p for p in cfg["phases"]}
    phase = cfg["phases"][0]
    overlap_pct = cfg["noise"].get("stage_overlap_pct", 0.0)
    blend_alpha = cfg["noise"].get("blend_alpha", 0.3)
    benign_ref = _make_benign_reference_row(cfg, rng) if overlap_pct > 0 else None

    rows = []
    t0 = datetime.now(timezone.utc)
    windows_emitted = 0
    dwell_remaining = int(rng.integers(phase["dwell_windows"][0], phase["dwell_windows"][1] + 1))

    while windows_emitted < total_windows:
        if dwell_remaining <= 0:
            nxt = phase.get("next")
            if nxt is None:
                # persona has run its course (e.g. one-shot kill chain) —
                # settle into idle/benign dwell for the remainder of the run
                if "idle" in phases_by_name:
                    phase = phases_by_name["idle"]
                else:
                    phase = phases_by_name[cfg["phases"][0]["name"]]
            else:
                phase = phases_by_name[nxt]
            dwell_remaining = int(rng.integers(phase["dwell_windows"][0], phase["dwell_windows"][1] + 1))

        feat = sample_window(cfg, phase, rng)
        if benign_ref is not None and rng.random() < overlap_pct:
            feat = blend_toward_benign(feat, benign_ref, blend_alpha)

        ts = (t0 + timedelta(seconds=WINDOW_SECONDS * windows_emitted)).isoformat()
        row = {
            "timestamp": ts,
            "persona_id": cfg["persona_id"],
            "run_id": run_id,
            "mitre_stage": phase["mitre_stage"],
            "technique_id": phase.get("technique_id") or cfg.get("mitre_technique") or "",
        }
        row.update(feat)
        row["_phase_name"] = phase["name"]  # internal only; write_csv filters this out via CSV_COLUMNS
        rows.append(row)

        windows_emitted += 1
        dwell_remaining -= 1

    return run_id, rows


def write_csv(rows: list, persona_id: str, run_id: str, out_root: str = "data/raw/personas") -> Path:
    out_dir = Path(out_root) / persona_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_id}.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r[k] for k in CSV_COLUMNS})
    return out_path
