"""Ingest pre-aggregated 32-d persona CSVs into the world-model pair format.

Persona CSVs are already windowed (5s cadence). This module does NOT go through
cic_schema alias-binding or CIC Label → MITRE mapping. The generator writes
STATE_FEATURE_ORDER columns and a STAGE_NAMES mitre_stage directly.

Shared interface (also in the persona-dev prompt). Do not change column names
or order without updating both sides first.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.world_model.cic_schema import INPUT_DIM, STATE_FEATURE_ORDER
from src.world_model.dataset import (
    DAY_ID_DTYPE,
    SOURCE_DTYPE,
    concat_window_bundles,
    make_synth_day_id,
)
from src.world_model.labels import STAGE_ID, STAGE_NAMES, technique_for_stage
from src.world_model.windows import WindowConfig

PERSONA_META_COLUMNS = (
    "timestamp",
    "persona_id",
    "run_id",
    "mitre_stage",
    "technique_id",
)

# Exact contract column order: 5 meta + 32 features matching STATE_FEATURE_ORDER.
PERSONA_CSV_COLUMNS = PERSONA_META_COLUMNS + tuple(STATE_FEATURE_ORDER)

OPTIONAL_HOST_COLUMNS = ("src_host", "dst_host")

MAX_GAP_SECONDS = 15.0


def _norm_header(name: str) -> str:
    return str(name).strip()


def validate_persona_columns(columns: list[str]) -> list[str]:
    """Require the 5 meta + 32 STATE_FEATURE_ORDER names. Extra columns allowed.

    If the 32 feature *names* match, they are reordered to STATE_FEATURE_ORDER.
    A name mismatch is a hard error (tell the persona dev; do not guess).
    """
    got = [_norm_header(c) for c in columns]
    got_set = set(got)
    missing_meta = [c for c in PERSONA_META_COLUMNS if c not in got_set]
    missing_feat = [c for c in STATE_FEATURE_ORDER if c not in got_set]
    if missing_meta or missing_feat:
        extra = [c for c in got if c not in PERSONA_CSV_COLUMNS and c not in OPTIONAL_HOST_COLUMNS]
        raise ValueError(
            "Persona CSV columns do not match the shared interface.\n"
            f"  missing meta: {missing_meta or 'none'}\n"
            f"  missing features: {missing_feat or 'none'}\n"
            f"  extra (ignored unless src_host/dst_host): {extra or 'none'}\n"
            f"  expected 32 features (STATE_FEATURE_ORDER): {list(STATE_FEATURE_ORDER)}\n"
            "Send this mismatch to the persona dev — do not invent aliases."
        )
    feat_in_file = [c for c in got if c in set(STATE_FEATURE_ORDER)]
    if feat_in_file != list(STATE_FEATURE_ORDER):
        print(
            "[persona] feature column order differs from STATE_FEATURE_ORDER; "
            "reordering by name (names themselves match)."
        )
    return got


def load_persona_csv(
    path: str | Path,
    *,
    window_seconds: int = 5,
    horizon_k: int = 6,
    max_gap_seconds: float = MAX_GAP_SECONDS,
) -> list[dict[str, Any]]:
    """Load one already-windowed persona CSV → list of (S_t, S_{t+1}) pair dicts.

    Groups by (persona_id, run_id). Each group is one synthetic 'day'.
    Keeps a pair only if the adjacent timestamp gap is in (0, max_gap_seconds].
    """
    path = Path(path)
    df = pd.read_csv(path)
    df.columns = [_norm_header(c) for c in df.columns]
    validate_persona_columns(list(df.columns))
    if df.empty:
        raise ValueError(f"Persona CSV is empty: {path}")

    ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    if float(ts.notna().mean()) < 0.9:
        raise ValueError(
            f"Unparseable timestamps in {path.name} "
            f"({float(ts.notna().mean()):.1%} valid ISO8601)."
        )
    work = df.loc[ts.notna()].copy()
    work["_ts"] = ts[ts.notna()]
    work["persona_id"] = work["persona_id"].astype(str).str.strip()
    work["run_id"] = work["run_id"].astype(str).str.strip()
    work["mitre_stage"] = work["mitre_stage"].astype(str).str.strip()
    tech_col = work["technique_id"] if "technique_id" in work.columns else None

    pairs: list[dict[str, Any]] = []
    for (persona_id, run_id), g in work.groupby(["persona_id", "run_id"], sort=True):
        if not persona_id or persona_id in {"nan", "None"}:
            raise ValueError(f"{path.name}: empty persona_id")
        if not run_id or str(run_id) in {"nan", "None"}:
            raise ValueError(f"{path.name}: empty run_id for persona {persona_id}")
        grp = g.sort_values("_ts")
        unknown = sorted({s for s in grp["mitre_stage"] if s not in STAGE_ID})
        if unknown:
            raise ValueError(
                f"{path.name} ({persona_id}/{run_id}): mitre_stage not in STAGE_NAMES: "
                f"{unknown}. Allowed: {list(STAGE_NAMES)}"
            )
        feat = grp[list(STATE_FEATURE_ORDER)].apply(pd.to_numeric, errors="coerce")
        if feat.isna().any().any():
            n_bad = int(feat.isna().any(axis=1).sum())
            raise ValueError(
                f"{path.name} ({persona_id}/{run_id}): {n_bad} rows have non-numeric features"
            )
        states = feat.to_numpy(dtype=np.float32)
        if states.shape[1] != INPUT_DIM:
            raise ValueError(f"expected {INPUT_DIM} features, got {states.shape[1]}")
        if not np.isfinite(states).all():
            raise ValueError(f"{path.name} ({persona_id}/{run_id}): non-finite feature values")
        unix = grp["_ts"].map(lambda t: pd.Timestamp(t).timestamp()).to_numpy(dtype=np.float64)
        stages = np.array([STAGE_ID[s] for s in grp["mitre_stage"]], dtype=np.int8)
        if tech_col is not None:
            techniques = [
                "" if pd.isna(v) else str(v).strip()
                for v in grp["technique_id"].tolist()
            ]
        else:
            techniques = [""] * len(grp)
        t0 = unix[0]
        wids = np.floor((unix - t0) / float(window_seconds)).astype(np.int64)
        pairs.extend(
            _pair_run(
                states=states,
                unix=unix,
                wids=wids,
                stages=stages,
                techniques=techniques,
                persona_id=str(persona_id),
                run_id=str(run_id),
                source_file=path.name,
                window_seconds=window_seconds,
                horizon_k=horizon_k,
                max_gap_seconds=max_gap_seconds,
            )
        )
    if not pairs:
        raise ValueError(f"No consecutive window pairs in {path}")
    return pairs


def _pair_run(
    *,
    states: np.ndarray,
    unix: np.ndarray,
    wids: np.ndarray,
    stages: np.ndarray,
    techniques: list[str],
    persona_id: str,
    run_id: str,
    source_file: str,
    window_seconds: int,
    horizon_k: int,
    max_gap_seconds: float,
) -> list[dict[str, Any]]:
    n = len(states)
    day_id = make_synth_day_id(persona_id, run_id)
    attack_now = (stages != STAGE_ID["benign"]).astype(np.int8)
    infil_now = (stages == STAGE_ID["lateral_movement"]).astype(np.int8)
    out: list[dict[str, Any]] = []
    max_wid_hole = max_gap_seconds / float(window_seconds)
    for i in range(n - 1):
        gap = float(unix[i + 1] - unix[i])
        if gap <= 0:
            continue
        wid_gap = int(wids[i + 1] - wids[i])
        if gap > max_gap_seconds + 1e-6 and wid_gap > max_wid_hole:
            continue
        future = []
        for j in range(i + 1, n):
            if unix[j] - unix[i] > horizon_k * window_seconds + 1e-6:
                break
            if wids[j] - wids[i] > horizon_k:
                break
            future.append(j)
        atk_k = int(any(attack_now[j] for j in future))
        inf_k = int(any(infil_now[j] for j in future))
        now = int(attack_now[i])
        tid = techniques[i]
        if not tid:
            tid, _ = technique_for_stage(int(stages[i]))
        out.append({
            "state": states[i],
            "next_state": states[i + 1],
            "attack_now": now,
            "attack_within_k": atk_k,
            "infiltration_within_k": inf_k,
            "pre_attack": int(atk_k == 1 and now == 0),
            "stage_id": int(stages[i]),
            "technique_id": tid,
            "timestamp": float(unix[i]),
            "window_id": int(wids[i]),
            "day_id": day_id,
            "source_file": source_file,
            "source": "synthetic",
            "persona_id": persona_id,
            "run_id": str(run_id),
        })
    return out


def pairs_to_bundle(pairs: list[dict[str, Any]], cfg: WindowConfig | None = None) -> dict:
    """Stack pair dicts into the same NPZ window-bundle format as windows.py."""
    if not pairs:
        raise ValueError("No persona pairs to bundle")
    cfg = cfg or WindowConfig()
    return {
        "states": np.stack([p["state"] for p in pairs]).astype(np.float32),
        "next_states": np.stack([p["next_state"] for p in pairs]).astype(np.float32),
        "attack_now": np.array([p["attack_now"] for p in pairs], dtype=np.int8),
        "attack_within_k": np.array([p["attack_within_k"] for p in pairs], dtype=np.int8),
        "infiltration_within_k": np.array([p["infiltration_within_k"] for p in pairs], dtype=np.int8),
        "pre_attack": np.array([p["pre_attack"] for p in pairs], dtype=np.int8),
        "stage_id": np.array([p["stage_id"] for p in pairs], dtype=np.int8),
        "technique_id": np.array([p["technique_id"] for p in pairs], dtype=object),
        "timestamps": np.array([p["timestamp"] for p in pairs], dtype=np.float64),
        "window_ids": np.array([p["window_id"] for p in pairs], dtype=np.int64),
        "day_id": np.array([p["day_id"] for p in pairs], dtype=DAY_ID_DTYPE),
        "source_file": np.array([p["source_file"] for p in pairs], dtype=object),
        "source": np.array([p["source"] for p in pairs], dtype=SOURCE_DTYPE),
        "persona_id": np.array([p["persona_id"] for p in pairs], dtype=DAY_ID_DTYPE),
        "run_id": np.array([p["run_id"] for p in pairs], dtype=DAY_ID_DTYPE),
        "feature_names": np.array(STATE_FEATURE_ORDER),
        "window_seconds": np.array([cfg.window_seconds]),
        "horizon_k": np.array([cfg.horizon_k]),
    }


def discover_persona_csvs(root: Path) -> list[Path]:
    """data/raw/personas/<persona_id>/<run_id>.csv — skip configs / hidden files."""
    if not root.is_dir():
        return []
    files = sorted(
        p for p in root.rglob("*.csv")
        if p.is_file() and not p.name.startswith(".")
    )
    return files


def load_persona_dir(root: str | Path, cfg: WindowConfig | None = None) -> dict | None:
    """Load every persona CSV under root and concat into one window bundle."""
    cfg = cfg or WindowConfig()
    paths = discover_persona_csvs(Path(root))
    if not paths:
        return None
    bundles = []
    for path in paths:
        print(f"[persona] {path}")
        pairs = load_persona_csv(path, window_seconds=cfg.window_seconds, horizon_k=cfg.horizon_k)
        bundle = pairs_to_bundle(pairs, cfg)
        print(
            f"         pairs={len(bundle['states'])}  "
            f"days={sorted(set(map(str, bundle['day_id'])))}  "
            f"attack_now={float(bundle['attack_now'].mean()):.3f}"
        )
        bundles.append(bundle)
    return concat_window_bundles(bundles)


# ---------------------------------------------------------------------------
# Schema-faithful fixtures (stand-in until the 20-persona generator lands)
# ---------------------------------------------------------------------------

# (persona_id, family/stage, technique_id, n_windows)
_FIXTURE_PERSONAS: tuple[tuple[str, str, str], ...] = (
    ("benign_office_v1", "benign", ""),
    ("recon_portscan_v1", "reconnaissance", "T1595.001"),
    ("ssh_bruteforce_v1", "initial_access", "T1110"),
    ("evasive_lateral_v1", "lateral_movement", "T1021"),
    ("bot_c2_beacon_v1", "command_and_control", "T1071"),
    ("exfil_dns_v1", "exfiltration", "T1048"),
    ("impact_flood_v1", "impact", "T1498"),
)

_RUN_IDS = ("run_000", "run_001", "run_002")


def _stage_vector(stage: str, rng: np.random.Generator, attack: bool) -> np.ndarray:
    """Hand-tuned 32-d profiles. Dims 29–31 are nonzero; not a single-feature tell."""
    v = np.zeros(32, dtype=np.float32)
    v[0] = float(rng.integers(8, 40))          # flow_count
    v[1] = float(rng.integers(3, 12))          # unique_dst_ports
    v[2] = float(rng.uniform(0.4, 1.8))        # dest_port_entropy
    v[3] = float(rng.uniform(0.7, 1.0))        # tcp_ratio
    v[4] = 1.0 - v[3]                          # udp_ratio
    v[5] = float(rng.uniform(800, 8_000))      # bytes_fwd
    v[6] = float(rng.uniform(400, 6_000))      # bytes_bwd
    tot = v[5] + v[6]
    v[7] = v[6] / tot if tot else 0.5
    v[8] = float(rng.integers(10, 80))
    v[9] = float(rng.integers(8, 70))
    v[10] = float(rng.uniform(1e4, 8e4))
    v[11] = float(rng.uniform(v[10], 2e5))
    v[12] = float(rng.uniform(200, 4000))
    v[13] = float(rng.uniform(50, 800))
    v[14] = float(rng.uniform(v[12], 2e4))
    v[15] = float(rng.integers(0, 6))
    v[16] = float(rng.integers(4, 30))
    v[17] = float(rng.integers(0, 4))
    v[18] = float(rng.integers(0, 3))
    v[19] = float(rng.integers(0, 8))
    v[20] = 0.0
    v[21] = float(rng.uniform(0.4, 1.6))
    v[22] = float(rng.uniform(80, 600))
    v[23] = float(rng.uniform(10, 120))
    v[24] = float(rng.uniform(200, 1460))
    v[25] = float(rng.choice([8192, 65535, 29200]))
    v[26] = float(rng.choice([8192, 65535, 28960]))
    v[27] = float(rng.integers(0, 6))
    v[28] = float(rng.uniform(0.0, 0.15))
    # Dims 29–31: real variance, present on benign too (so they are not a label leak).
    v[29] = float(rng.uniform(0.2, 2.5))       # ttl_variance
    v[30] = float(rng.integers(0, 2))          # ip_fragment_flags
    v[31] = float(rng.integers(0, 3))          # retransmit_count

    if not attack or stage == "benign":
        return v

    if stage == "reconnaissance":
        v[1] = float(rng.integers(24, 60))
        v[2] = float(rng.uniform(3.0, 5.5))
        v[15] = float(rng.integers(20, 80))
        v[28] = float(rng.uniform(0.55, 0.95))
        v[3] = 1.0
        v[4] = 0.0
        v[5] = float(rng.uniform(200, 1200))
        v[6] = float(rng.uniform(80, 400))
    elif stage == "initial_access":
        v[1] = 1.0
        v[2] = 0.0
        v[15] = float(rng.integers(8, 40))
        v[16] = float(rng.integers(8, 40))
        v[19] = float(rng.integers(4, 20))
        v[25] = 65535.0
    elif stage == "lateral_movement":
        v[1] = float(rng.integers(4, 14))
        v[5] = float(rng.uniform(2e3, 2e4))
        v[8] = float(rng.integers(20, 120))
        v[29] = float(rng.uniform(4.0, 12.0))   # evasive TTL jitter — not the only cue
        v[30] = float(rng.integers(1, 6))
        v[31] = float(rng.integers(1, 8))
    elif stage == "command_and_control":
        v[0] = float(rng.integers(4, 12))
        v[1] = 1.0
        v[2] = 0.0
        v[12] = float(rng.uniform(4e4, 9e4))    # beacon-like IAT
        v[13] = float(rng.uniform(50, 400))
        v[5] = float(rng.uniform(200, 900))
        v[6] = float(rng.uniform(150, 800))
        v[31] = float(rng.integers(2, 10))
    elif stage == "exfiltration":
        v[5] = float(rng.uniform(5e4, 4e5))
        v[6] = float(rng.uniform(200, 2000))
        v[7] = v[6] / (v[5] + v[6])
        v[8] = float(rng.integers(80, 400))
        v[4] = float(rng.uniform(0.4, 0.9))
        v[3] = 1.0 - v[4]
    elif stage == "impact":
        v[0] = float(rng.integers(80, 250))
        v[8] = float(rng.integers(200, 800))
        v[15] = float(rng.integers(40, 200))
        v[5] = float(rng.uniform(1e4, 8e4))
        v[28] = float(rng.uniform(0.2, 0.6))
    return v


def generate_persona_fixture_frame(
    persona_id: str,
    run_id: str,
    stage: str,
    technique_id: str,
    *,
    n_windows: int = 48,
    t0: datetime | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t0 = t0 or datetime(2026, 6, 1, tzinfo=timezone.utc)
    warmup = 8 if stage != "benign" else 0
    rows = []
    for w in range(n_windows):
        attack = stage != "benign" and w >= warmup
        vec = _stage_vector(stage, rng, attack)
        row: dict[str, Any] = {
            "timestamp": (t0 + timedelta(seconds=5 * w)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "persona_id": persona_id,
            "run_id": run_id,
            "mitre_stage": stage if attack else "benign",
            "technique_id": technique_id if attack else "",
        }
        for i, name in enumerate(STATE_FEATURE_ORDER):
            row[name] = float(vec[i])
        rows.append(row)
    return pd.DataFrame(rows, columns=list(PERSONA_CSV_COLUMNS))


def write_schema_fixtures(
    root: str | Path,
    *,
    n_windows: int = 48,
    n_runs: int = 3,
) -> list[Path]:
    """Write 7-stage × n_runs CSVs under root/<persona_id>/<run_id>.csv."""
    root = Path(root)
    written: list[Path] = []
    run_ids = _RUN_IDS[: max(1, n_runs)]
    for p_i, (persona_id, stage, tid) in enumerate(_FIXTURE_PERSONAS):
        for r_i, run_id in enumerate(run_ids):
            t0 = datetime(2026, 6, 1, tzinfo=timezone.utc) + timedelta(days=p_i, hours=r_i)
            seed = 1000 * p_i + 17 * r_i + 7
            df = generate_persona_fixture_frame(
                persona_id, run_id, stage, tid, n_windows=n_windows, t0=t0, seed=seed,
            )
            out = root / persona_id / f"{run_id}.csv"
            out.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(out, index=False)
            written.append(out)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Persona CSV ingest / schema fixtures")
    parser.add_argument("--write-fixtures", metavar="DIR", help="Write 7-stage fixture CSVs")
    parser.add_argument("--n-windows", type=int, default=48)
    parser.add_argument("--n-runs", type=int, default=3)
    args = parser.parse_args()
    if args.write_fixtures:
        paths = write_schema_fixtures(args.write_fixtures, n_windows=args.n_windows, n_runs=args.n_runs)
        print(f"[fixtures] wrote {len(paths)} CSVs under {args.write_fixtures}")
        return
    raise SystemExit("Pass --write-fixtures DIR")


if __name__ == "__main__":
    main()
