"""
Production dataset build.

Directly responds to the pipeline dev's report finding: fixture-based C2
(39 windows) didn't transfer to real 02 Mar C2 traffic and didn't even fire
on held-out fixture C2. That's a mass problem, not a signature problem —
so this script deliberately gives reconnaissance / command_and_control /
exfiltration (and the evasive variants that also emit those stages) enough
runs to land in the same order of magnitude as the smallest real CIC class
(impact: 1,305 windows; lateral_movement: 1,635 windows) instead of 39.

Run-id convention matches the report's TRAIN/VAL/TEST split scheme:
run_000 = train, run_001 = val, run_002 = test, run_003+ = extra train mass.
A run is never split internally (same rule as CIC calendar days).

Usage:
    python3 produce_dataset.py [--out-dir data/raw/personas] [--clean]
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

from generator.config_io import load_all_configs
from generator.engine import generate_run, write_csv

# (runs, windows_per_run) per tier. windows=200 => ~1000s of simulated
# seconds per run, well above the 8-window score_history context length.
TIER1_DATASET_CALIBRATED = {
    "benign_browsing_v1", "benign_bulk_transfer_v1", "benign_iot_idle_v1",
    "bruteforce_ssh_v1", "bruteforce_ftp_v1", "web_bruteforce_v1",
    "dos_hulk_v1", "dos_slowloris_v1",
    "web_sql_injection_v1", "web_xss_v1",
    "infiltration_v1", "lateral_movement_smb_v1",
}
TIER1_PLAN = {"runs": 3, "windows": 720}   # ~2,160 windows/persona — CIC already carries these classes' bulk

TIER2_MITRE_DOC_BASED = {
    "recon_portscan_v1", "recon_service_discovery_v1",
    "c2_beacon_v1", "c2_dns_tunnel_v1",
    "exfil_c2channel_v1",
}
TIER2_PLAN = {"runs": 8, "windows": 200}   # ~1,600 windows/persona — the mass tier2 was missing

TIER3_ADAPTIVE = {
    "evasive_recon_v1", "evasive_c2_v1", "evasive_lateral_v1",
}
TIER3_PLAN = {"runs": 5, "windows": 300}   # ~1,500 windows/persona


def plan_for(pid):
    if pid in TIER2_MITRE_DOC_BASED:
        return TIER2_PLAN
    if pid in TIER3_ADAPTIVE:
        return TIER3_PLAN
    return TIER1_PLAN


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-dir", type=str, default="personas/configs")
    ap.add_argument("--out-dir", type=str, default="data/raw/personas")
    ap.add_argument("--clean", action="store_true", help="wipe out-dir before generating")
    args = ap.parse_args()

    if args.clean:
        import shutil
        shutil.rmtree(args.out_dir, ignore_errors=True)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    cfgs = load_all_configs(args.config_dir)
    manifest = []
    stage_totals = defaultdict(int)

    for pid, cfg in sorted(cfgs.items()):
        plan = plan_for(pid)
        persona_stage_counts = defaultdict(int)
        n_runs, n_windows = plan["runs"], plan["windows"]

        for run_idx in range(n_runs):
            run_id = f"{pid}__run_{run_idx:03d}"
            run_id, rows = generate_run(cfg, total_windows=n_windows, run_id=run_id, seed_offset=run_idx * 100003)
            write_csv(rows, pid, run_id, out_root=args.out_dir)
            for r in rows:
                persona_stage_counts[r["mitre_stage"]] += 1
                stage_totals[r["mitre_stage"]] += 1

        total = sum(persona_stage_counts.values())
        manifest.append({
            "persona_id": pid, "family": cfg["family"], "adaptive": cfg["adaptive"],
            "runs": n_runs, "windows_per_run": n_windows, "total_windows": total,
            "stage_breakdown": dict(persona_stage_counts),
            "split": {"train": [f"run_{i:03d}" for i in range(n_runs) if i != 1 and i != 2],
                      "val": ["run_001"] if n_runs > 1 else [],
                      "test": ["run_002"] if n_runs > 2 else []},
        })

    with open("docs/dataset_manifest.json", "w") as f:
        json.dump({"personas": manifest, "stage_totals": dict(stage_totals)}, f, indent=2)

    print(f"{'persona_id':32s} {'runs':>5s} {'win/run':>8s} {'total':>7s}")
    for m in manifest:
        print(f"{m['persona_id']:32s} {m['runs']:5d} {m['windows_per_run']:8d} {m['total_windows']:7d}")
    print("-" * 60)
    print("Aggregate windows per mitre_stage (synthetic side only, not combined with CIC):")
    for stage, n in sorted(stage_totals.items(), key=lambda kv: -kv[1]):
        print(f"  {stage:22s} {n:7d}")
    print()
    print("Manifest written to docs/dataset_manifest.json")


if __name__ == "__main__":
    main()
