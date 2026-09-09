"""
One-time sanity check for dataset-calibrated personas: compare generated
mean/std of a handful of key features against reference per-class stats,
report a z-score per feature. This is deliberately NOT an iterative loop —
run once after the generator works, per the brief.

NOTE: personas/reference_stats.json currently ships with PLACEHOLDER values
(see that file's warning field). Swap it for the pipeline dev's real
CIC-IDS2018 per-class mean/std/max numbers before treating |z| here as a
genuine synthetic-vs-real divergence signal.
"""
import json
from pathlib import Path

import pandas as pd

from generator.config_io import load_all_configs

KEY_FEATURES = [
    "flow_count", "syn_flag_sum", "dest_port_entropy",
    "down_up_ratio_mean", "pkt_len_mean", "port_scan_score", "retransmit_count",
]

# Only the dataset-calibrated tier gets checked against dataset-derived stats —
# the MITRE-doc-based and adaptive personas have no real-data analogue by design.
DATASET_CALIBRATED_PERSONAS = {
    "benign_browsing_v1", "benign_bulk_transfer_v1", "benign_iot_idle_v1",
    "bruteforce_ssh_v1", "bruteforce_ftp_v1", "web_bruteforce_v1",
    "dos_hulk_v1", "dos_slowloris_v1",
    "web_sql_injection_v1", "web_xss_v1",
    "infiltration_v1", "lateral_movement_smb_v1",
}


def load_all_runs_for(persona_id, data_root="data/raw/personas"):
    d = Path(data_root) / persona_id
    files = sorted(d.glob("*.csv"))
    assert files, f"no generated CSV found for {persona_id} — run produce_dataset.py first"
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def main():
    with open("personas/reference_stats.json") as f:
        ref = json.load(f)
    warning = ref.pop("_PLACEHOLDER_WARNING", None)

    cfgs = load_all_configs("personas/configs")
    rows_out = []

    for pid in sorted(DATASET_CALIBRATED_PERSONAS):
        cfg = cfgs[pid]
        family = cfg["phases"][0]["mitre_stage"]
        if family not in ref:
            continue
        df = load_all_runs_for(pid)

        for feat in KEY_FEATURES:
            gen_mean = df[feat].mean()
            gen_std = df[feat].std()
            r = ref[family][feat]
            z = (gen_mean - r["mean"]) / r["std"] if r["std"] > 0 else float("nan")
            rows_out.append({
                "persona_id": pid, "family": family, "feature": feat,
                "generated_mean": round(gen_mean, 3), "generated_std": round(gen_std, 3),
                "reference_mean": r["mean"], "reference_std": r["std"],
                "z_score": round(z, 2),
                "flag": "REVIEW" if abs(z) > 2 else "ok",
            })

    report = pd.DataFrame(rows_out)
    out_path = Path("docs/divergence_report.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(out_path, index=False)

    print("=" * 78)
    if warning:
        print("PLACEHOLDER REFERENCE STATS IN USE:")
        print(warning)
        print("=" * 78)
    print(report.to_string(index=False))
    n_flag = (report["flag"] == "REVIEW").sum()
    print("=" * 78)
    print(f"{n_flag} / {len(report)} feature checks flagged (|z| > 2). Full table: {out_path}")


if __name__ == "__main__":
    main()
