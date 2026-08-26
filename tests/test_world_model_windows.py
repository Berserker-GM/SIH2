#!/usr/bin/env python3
"""Tests for CIC-IDS state-window construction (SIH world-model data layer)."""

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.world_model.cic_schema import INPUT_DIM, PACKET_LEVEL_SPEC, STATE_FEATURE_ORDER, bind_columns
from src.world_model.labels import STAGE_ID, map_label
from src.world_model.prepare_dataset import generate_demo_flows
from src.world_model.windows import WindowConfig, build_state_windows

results = []


def check(name: str, condition: bool, detail: str = "") -> None:
    results.append((name, condition))
    icon = "PASS" if condition else "FAIL"
    extra = f"  ({detail})" if detail else ""
    print(f"  {icon}  {name}{extra}")


print("\n-- Label mapping ------------------------------------------------------")
ssh = map_label("SSH-Bruteforce")
check("SSH brute → initial_access", ssh["stage"] == "initial_access")
check("SSH brute → T1110", ssh["technique_id"] == "T1110")
check("SSH brute is_attack", ssh["is_attack"] is True)

inf = map_label("Infilteration")  # CIC-IDS2018 spelling
check("Infilteration → lateral_movement", inf["stage"] == "lateral_movement")
check("Infilteration is_infiltration", inf["is_infiltration"] is True)

ben = map_label("Benign")
check("Benign is not attack", ben["is_attack"] is False)
check("Benign stage_id = 0", ben["stage_id"] == STAGE_ID["benign"])

hulk = map_label("DoS attacks-Hulk")
check("DoS Hulk → impact", hulk["stage"] == "impact")


print("\n-- Column aliases -----------------------------------------------------")
bound = bind_columns([" Dst Port", "Label", "Flow Duration", "Timestamp"])
check("binds Dst Port", bound.get("dst_port") == " Dst Port")
check("binds Label", bound.get("label") == "Label")


print("\n-- Window construction ------------------------------------------------")
df = generate_demo_flows(n_windows=24, flows_per_window=6)
check("demo frame has Label", "Label" in df.columns)
check("demo frame has Timestamp", "Timestamp" in df.columns)

bundle = build_state_windows(df, WindowConfig(window_seconds=5, horizon_k=6))
check("input dim = 32", bundle["states"].shape[1] == INPUT_DIM)
check("next_states aligned", bundle["states"].shape == bundle["next_states"].shape)
check("at least 10 pairs", bundle["states"].shape[0] >= 10)
check("feature_names length", len(bundle["feature_names"]) == len(STATE_FEATURE_ORDER))
check(
    "packet-level order 22–31",
    [row["name"] for row in PACKET_LEVEL_SPEC] == list(STATE_FEATURE_ORDER[22:]),
)
ttl_idx = STATE_FEATURE_ORDER.index("ttl_variance")
check("PCAP stubs are zero", float(bundle["states"][:, ttl_idx].sum()) == 0.0)
check("states are finite", bool(np.isfinite(bundle["states"]).all()))
check("next_states are finite", bool(np.isfinite(bundle["next_states"]).all()))

early_attack_now = int(bundle["attack_now"][:6].sum())
later_attack = int(bundle["attack_now"][8:].sum())
lookahead = int(bundle["attack_within_k"][:8].sum())
check("early windows mostly benign", early_attack_now <= 1)
check("later windows contain attacks", later_attack >= 1)
check("lookahead labels future attack", lookahead >= 1)
check("infiltration lookahead present", int(bundle["infiltration_within_k"].sum()) >= 1)
check("pre_attack flag exists", "pre_attack" in bundle)
check("some benign windows precede attack", int(bundle["pre_attack"].sum()) >= 1)

print("\n-- Live loader (small files stay whole) -------------------------------")
from src.world_model.windows import load_cic_csv_live, _pick_live_interval
import pandas as pd

demo_path = PROJECT_ROOT / "data" / "processed" / "_demo_live.csv"
df.to_csv(demo_path, index=False)
live_df, live_meta = load_cic_csv_live(demo_path)
check("demo live is not truncated", live_meta.get("truncated") is False)
check("demo live row count", int(live_meta.get("n_rows") or 0) == len(df))
demo_path.unlink(missing_ok=True)

idx = pd.date_range("2018-02-20 10:00:00", periods=40, freq="5s", tz="UTC")
ts = pd.Series(np.repeat(idx, 2))
labels = pd.Series(["DDoS attacks-LOIC-HTTP"] * 40 + ["Benign"] * 40)
t0_latest, t1_latest, reason_latest = _pick_live_interval(ts, labels, 90.0)
check("default live interval is latest", "latest" in reason_latest)
check("latest slice is 90s", abs((t1_latest - t0_latest).total_seconds() - 90.0) < 1.0)
check("latest starts in benign tail", t0_latest >= idx[18])
t0_atk, t1_atk, reason_atk = _pick_live_interval(ts, labels, 90.0, mode="attack")
check("attack mode names attack window", "attack" in reason_atk)
check("attack mode is 90s", abs((t1_atk - t0_atk).total_seconds() - 90.0) < 1.0)
check("attack mode starts in attack half", t0_atk <= idx[20])


print("\n-- Summary ------------------------------------------------------------")
passed = sum(1 for _, ok in results if ok)
failed = sum(1 for _, ok in results if not ok)
print(f"\n  Total: {len(results)}   Passed: {passed}   Failed: {failed}\n")
if failed:
    sys.exit(1)
print("  World-model data layer is ready.\n")
