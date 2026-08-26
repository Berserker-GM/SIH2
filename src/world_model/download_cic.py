"""
Download CSE-CIC-IDS2018 processed CSVs from the official AWS Open Data bucket.

NCIIPC (nciipc.gov.in) does not host these files. SIH points teams at public
corpora; this script fetches the CICFlowMeter CSVs used for training.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

S3_PREFIX = (
    "https://cse-cic-ids2018.s3.ca-central-1.amazonaws.com/"
    "Processed%20Traffic%20Data%20for%20ML%20Algorithms/"
)

# Official daily ML CSVs. Default = infiltration day (SIH target scenario).
DATASETS = {
    "ftp-ssh-brute": "Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv",
    "dos-goldeneye-slowloris": "Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv",
    "dos-hulk-slowhttp": "Friday-16-02-2018_TrafficForML_CICFlowMeter.csv",
    "ddos-loic-http": "Tuesday-20-02-2018_TrafficForML_CICFlowMeter.csv",
    "ddos-hoic": "Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv",
    "web-22": "Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv",
    "web-23": "Friday-23-02-2018_TrafficForML_CICFlowMeter.csv",
    "infiltration": "Wednesday-28-02-2018_TrafficForML_CICFlowMeter.csv",
    "infiltration-03-01": "Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv",
    "botnet": "Friday-02-03-2018_TrafficForML_CICFlowMeter.csv",
}

UNB_CITATION = "https://www.unb.ca/cic/datasets/ids-2018.html"


def _progress(block: int, block_size: int, total: int) -> None:
    if total <= 0:
        return
    done = block * block_size
    pct = min(100.0, 100.0 * done / total)
    mb = done / (1024 * 1024)
    total_mb = total / (1024 * 1024)
    sys.stdout.write(f"\r  {mb:.1f}/{total_mb:.1f} MB ({pct:5.1f}%)")
    sys.stdout.flush()


def download(name: str, dest_dir: Path) -> Path:
    if name not in DATASETS:
        raise SystemExit(f"Unknown dataset '{name}'. Choose from: {', '.join(DATASETS)}")
    filename = DATASETS[name]
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / filename
    if dest.exists() and dest.stat().st_size > 1_000_000:
        print(f"[skip] already present: {dest} ({dest.stat().st_size} bytes)")
        return dest
    url = S3_PREFIX + filename
    print(f"[download] {url}")
    print(f"[dest]     {dest}")
    print(f"[cite]     {UNB_CITATION}")
    urllib.request.urlretrieve(url, dest, reporthook=_progress)
    print(f"\n[done] {dest.stat().st_size} bytes")
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description="Download CSE-CIC-IDS2018 flow CSVs")
    parser.add_argument("--dataset", default="infiltration", choices=sorted(DATASETS),
                        help="Which attack-day CSV to fetch (default: infiltration)")
    parser.add_argument("--out", default="data/raw", help="Destination directory")
    parser.add_argument("--list", action="store_true", help="List available day files")
    args = parser.parse_args()

    if args.list:
        for key, fname in DATASETS.items():
            print(f"  {key:28s}  {fname}")
        print(f"\nSource: {UNB_CITATION}")
        print("NCIIPC does not host a copy; this is the organizer-approved public corpus.")
        return

    root = Path(__file__).resolve().parents[2]
    dest_dir = Path(args.out)
    if not dest_dir.is_absolute():
        dest_dir = root / dest_dir
    download(args.dataset, dest_dir)


if __name__ == "__main__":
    main()
