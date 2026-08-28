#!/usr/bin/env python3
"""Persona CSV ingest, NPZ merge, and in-process score_history()."""

from __future__ import annotations

import inspect
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.world_model.cic_schema import INPUT_DIM, STATE_FEATURE_ORDER
from src.world_model.dataset import (
    concat_window_bundles,
    fit_scaler,
    is_synthetic_day,
    load_npz,
    make_synth_day_id,
    split_synthetic_days,
)
from src.world_model.explain import LSTM_ATTACK_THRESHOLD
from src.world_model.labels import STAGE_ID, STAGE_NAMES
from src.world_model.mitre_decode import fit_stage_decoder
from src.world_model.personas import (
    PERSONA_CSV_COLUMNS,
    generate_persona_fixture_frame,
    load_persona_csv,
    load_persona_dir,
    pairs_to_bundle,
    write_schema_fixtures,
)
from src.world_model.prepare_dataset import generate_demo_flows
from src.world_model.service import (
    WorldModelRuntime,
    _VERSIONED_RUNTIME,
    score_history,
)
from src.world_model.windows import WindowConfig, build_state_windows

results = []


def check(name: str, condition: bool, detail: str = "") -> None:
    results.append((name, condition))
    icon = "PASS" if condition else "FAIL"
    extra = f"  ({detail})" if detail else ""
    print(f"  {icon}  {name}{extra}")


print("\n-- Shared interface vs STATE_FEATURE_ORDER -----------------------------")
feat_cols = PERSONA_CSV_COLUMNS[5:]
check("32 feature columns", len(feat_cols) == 32 == len(STATE_FEATURE_ORDER))
check("names match STATE_FEATURE_ORDER exactly", list(feat_cols) == list(STATE_FEATURE_ORDER))
check("ttl_variance is dim 29", STATE_FEATURE_ORDER[29] == "ttl_variance")
check("retransmit_count is dim 31", STATE_FEATURE_ORDER[31] == "retransmit_count")
check("score_history signature", inspect.signature(score_history).parameters.keys() >= {"x", "model_version"})
sig = inspect.signature(score_history)
check("score_history default v2", sig.parameters["model_version"].default == "v2")


print("\n-- load_persona_csv pairing -------------------------------------------")
with tempfile.TemporaryDirectory() as td:
    t0 = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)
    df = generate_persona_fixture_frame(
        "ssh_bruteforce_v1", "run_000", "initial_access", "T1110",
        n_windows=24, t0=t0, seed=0,
    )
    path = Path(td) / "run_000.csv"
    df.to_csv(path, index=False)
    pairs = load_persona_csv(path)
    check("returns a list", isinstance(pairs, list) and len(pairs) >= 10)
    check("pair has 32-d state", pairs[0]["state"].shape == (INPUT_DIM,))
    check("source is synthetic", pairs[0]["source"] == "synthetic")
    check("day_id is syn:persona:run", pairs[0]["day_id"] == "syn:ssh_bruteforce_v1:run_000")
    check("stage_id from mitre_stage", pairs[0]["stage_id"] in set(STAGE_ID.values()))
    check("dims 29-31 not all zero", float(sum(abs(float(p["state"][29])) + abs(float(p["state"][30])) + abs(float(p["state"][31])) for p in pairs)) > 0)
    ts = [p["timestamp"] for p in pairs]
    gaps = np.diff(ts)
    check("adjacent gaps <= 15s", bool(np.all(gaps <= 15.0 + 1e-6)))
    check("strictly increasing", bool(np.all(gaps > 0)))

    # Two runs in one file must not pair across runs
    df2 = generate_persona_fixture_frame(
        "ssh_bruteforce_v1", "run_001", "initial_access", "T1110",
        n_windows=12, t0=t0, seed=1,
    )
    both = Path(td) / "both.csv"
    import pandas as pd
    pd.concat([df, df2], ignore_index=True).to_csv(both, index=False)
    mixed = load_persona_csv(both)
    days = {p["day_id"] for p in mixed}
    check("two run_ids stay two days", days == {"syn:ssh_bruteforce_v1:run_000", "syn:ssh_bruteforce_v1:run_001"})


print("\n-- Never glue two synthetic runs --------------------------------------")
ids = [
    make_synth_day_id("recon_portscan_v1", "run_000"),
    make_synth_day_id("recon_portscan_v1", "run_001"),
    make_synth_day_id("recon_portscan_v1", "run_002"),
    make_synth_day_id("bot_c2_beacon_v1", "run_000"),
]
tr, va, te = split_synthetic_days(ids)
check("3-run persona: train/val/test each get a run", len(tr) >= 1 and len(va) == 1 and len(te) == 1)
check("no run in two splits", len(set(tr) & set(va) & set(te)) == 0 and not (set(tr) & set(te)))
check("is_synthetic_day", is_synthetic_day(ids[0]) and not is_synthetic_day("2018-03-01"))


print("\n-- Merge with CIC NPZ format ------------------------------------------")
cic = build_state_windows(
    generate_demo_flows(n_windows=16, flows_per_window=3),
    WindowConfig(),
    source_file="demo.csv",
)
check("CIC pairs tagged real", str(cic["source"][0]) == "real")
with tempfile.TemporaryDirectory() as td:
    root = Path(td) / "personas"
    write_schema_fixtures(root, n_windows=20, n_runs=2)
    persona = load_persona_dir(root, WindowConfig())
    check("persona dir loaded", persona is not None and len(persona["states"]) > 50)
    combined = concat_window_bundles([cic, persona])
    check("concat dim 32", combined["states"].shape[1] == 32)
    check("concat has both sources", set(map(str, combined["source"])) == {"real", "synthetic"})
    check("keys match CIC bundle", set(cic.keys()) <= set(combined.keys()))
    npz_path = Path(td) / "combined.npz"
    np.savez_compressed(npz_path, **{k: v for k, v in combined.items() if not str(k).startswith("_")})
    loaded = load_npz(npz_path)
    check("load_npz keeps synthetic days", any(is_synthetic_day(d) for d in loaded["day_id"]))
    check("load_npz source field", "synthetic" in set(map(str, loaded["source"])))
    check("day_id not truncated", any(len(str(d)) > 10 for d in loaded["day_id"]))


print("\n-- Unknown mitre_stage is refused -------------------------------------")
with tempfile.TemporaryDirectory() as td:
    df = generate_persona_fixture_frame(
        "ssh_bruteforce_v1", "run_000", "initial_access", "T1110", n_windows=8,
    )
    df.loc[0, "mitre_stage"] = "not_a_stage"
    path = Path(td) / "bad.csv"
    df.to_csv(path, index=False)
    raised = False
    try:
        load_persona_csv(path)
    except ValueError as exc:
        raised = "STAGE_NAMES" in str(exc) or "mitre_stage" in str(exc)
    check("unknown stage raises", raised)


print("\n-- Column mismatch is a hard error ------------------------------------")
with tempfile.TemporaryDirectory() as td:
    df = generate_persona_fixture_frame(
        "ssh_bruteforce_v1", "run_000", "initial_access", "T1110", n_windows=8,
    )
    df = df.rename(columns={"ttl_variance": "ttl_var"})
    path = Path(td) / "mismatch.csv"
    df.to_csv(path, index=False)
    raised = False
    try:
        load_persona_csv(path)
    except ValueError as exc:
        raised = "ttl_variance" in str(exc)
    check("renamed feature raises with the real name", raised)


print("\n-- score_history in-process (no HTTP) ---------------------------------")


class CopyLast(nn.Module):
    def forward(self, x):
        last = x[:, -1, :]
        logit = 4.0 * (last[:, 15] > 1.0).float() - 2.0
        return last, logit


X = np.zeros((40, 32), dtype=np.float32)
y = np.zeros(40, dtype=np.int32)
X[20:, 15] = 5.0
y[20:] = STAGE_ID["initial_access"]
decoder = fit_stage_decoder(X, y)
from src.world_model.dataset import Scaler
scaler = Scaler(mean=np.zeros(32), std=np.ones(32))
rt = WorldModelRuntime(model=CopyLast(), scaler=scaler, decoder=decoder, examples=[])
_VERSIONED_RUNTIME["v2"] = rt
_VERSIONED_RUNTIME["v1"] = rt

hist = np.zeros((8, 32), dtype=np.float32)
hist[:, 15] = 5.0
out = score_history(hist, model_version="v2")
check("attack_probability is float", isinstance(out["attack_probability"], float))
check("something_bad is bool", isinstance(out["something_bad"], bool))
check("stage in STAGE_NAMES", out["stage"] in STAGE_NAMES)
check("technique_id str or None", out["technique_id"] is None or isinstance(out["technique_id"], str))
check("why_attack_top_features is list of tuples", isinstance(out["why_attack_top_features"], list))
if out["why_attack_top_features"]:
    name, score = out["why_attack_top_features"][0]
    check("tuple is (str, float)", isinstance(name, str) and isinstance(score, float))
check("ood_score present", "ood_score" in out and "ood_flag" in out)
check("threshold still 0.15", LSTM_ATTACK_THRESHOLD == 0.15)
bad_shape = False
try:
    score_history(np.zeros((4, 32)), model_version="v2")
except ValueError:
    bad_shape = True
check("rejects wrong history shape", bad_shape)
_VERSIONED_RUNTIME.clear()


print("\n-- Summary ------------------------------------------------------------")
passed = sum(1 for _, ok in results if ok)
failed = sum(1 for _, ok in results if not ok)
print(f"\n  Total: {len(results)}   Passed: {passed}   Failed: {failed}\n")
if failed:
    sys.exit(1)
print("  Persona ingest + score_history OK.\n")
