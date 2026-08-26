"""
train/train_iforest.py — Dev 2 (Omkar)
Fit the Isolation Forest on benign rows from CICIDS2018 (or UNSW-NB15)
and save the fitted scorer to: ml_models/models/iforest.pkl

Usage:
    python train/train_iforest.py --csv /path/to/cicids2018_benign.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from isolation_forest import IForestScorer, FEATURE_ORDER, MODEL_PATH


def load_benign_csv(csv_path: str) -> list[dict]:
    df = pd.read_csv(csv_path, low_memory=False)
    df.columns = df.columns.str.strip()

    for label_col in ("Label", "label", "Class", "class"):
        if label_col in df.columns:
            df = df[df[label_col].str.upper() == "BENIGN"].copy()
            break

    missing = [f for f in FEATURE_ORDER if f not in df.columns]
    if missing:
        print(f"[WARN] Missing columns (will be zero-filled): {missing}")
        for m in missing:
            df[m] = 0.0

    records = df[FEATURE_ORDER].fillna(0).to_dict(orient="records")
    print(f"[DATA] Loaded {len(records)} benign rows from {csv_path}")
    return records


def train(csv_path: str) -> None:
    records = load_benign_csv(csv_path)

    scorer = IForestScorer()
    scorer.fit(records)
    scorer.save(MODEL_PATH)

    # Quick sanity check: score first 10 rows
    print("\n[SANITY] Scores on first 10 benign samples (should be low):")
    for i, fv in enumerate(records[:10]):
        s = scorer.score(fv)
        print(f"  sample {i:2d}: iforest_score={s:.4f}")

    print(f"\n[DONE] IForest fitted and saved → {MODEL_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit Isolation Forest on benign traffic")
    parser.add_argument("--csv", required=True, help="Path to CICIDS2018 / UNSW-NB15 CSV")
    args = parser.parse_args()
    train(args.csv)
