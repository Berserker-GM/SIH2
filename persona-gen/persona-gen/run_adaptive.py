"""
Run the (1+1) hill-climbing search for every adaptive persona.

Deliberately run pre-demo / offline (see brief: "hill-climbing convergence
is stochastic and you don't want to gamble that on stage").

Usage:
    python3 run_adaptive.py [--cycles N] [--persona persona_id]
"""
import argparse

from generator.adaptive import run_hill_climb
from generator.config_io import load_all_configs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycles", type=int, default=20)
    ap.add_argument("--persona", type=str, default=None)
    ap.add_argument("--config-dir", type=str, default="personas/configs")
    args = ap.parse_args()

    cfgs = load_all_configs(args.config_dir)
    adaptive_cfgs = {pid: c for pid, c in cfgs.items() if c["adaptive"]}
    if args.persona:
        adaptive_cfgs = {args.persona: adaptive_cfgs[args.persona]}

    for pid, cfg in adaptive_cfgs.items():
        summary = run_hill_climb(cfg, cycles=args.cycles)
        drop = summary["baseline_attack_probability"] - summary["final_attack_probability"]
        print(f"{pid:24s} baseline={summary['baseline_attack_probability']:.4f} "
              f"-> final={summary['final_attack_probability']:.4f}  "
              f"(drop={drop:+.4f})  log={summary['log_path']}")


if __name__ == "__main__":
    main()
