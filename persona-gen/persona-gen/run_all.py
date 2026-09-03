"""
Generate one run of traffic for every persona in personas/configs/.

Usage:
    python3 run_all.py [--windows N] [--persona persona_id]

Each invocation mints a fresh run_id per persona (see interface contract:
never let two runs' rows blend into one sequence).
"""
import argparse
from pathlib import Path

from generator.config_io import load_all_configs
from generator.engine import generate_run, write_csv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", type=int, default=720, help="windows per run (720 = 1 simulated hour at 5s stride)")
    ap.add_argument("--persona", type=str, default=None, help="generate only this persona_id")
    ap.add_argument("--config-dir", type=str, default="personas/configs")
    ap.add_argument("--out-dir", type=str, default="data/raw/personas")
    args = ap.parse_args()

    cfgs = load_all_configs(args.config_dir)
    if args.persona:
        cfgs = {args.persona: cfgs[args.persona]}

    for pid, cfg in cfgs.items():
        run_id, rows = generate_run(cfg, total_windows=args.windows)
        out_path = write_csv(rows, pid, run_id, out_root=args.out_dir)
        stage_counts = {}
        for r in rows:
            stage_counts[r["mitre_stage"]] = stage_counts.get(r["mitre_stage"], 0) + 1
        print(f"{pid:32s} run_id={run_id:40s} rows={len(rows):5d} -> {out_path}  stages={stage_counts}")


if __name__ == "__main__":
    main()
