# Dataset layer (SIH World Model)

NCIIPC (`nciipc.gov.in` / `helpdesk1@nciipc.gov.in`) does **not** host a downloadable traffic corpus. The problem statement tells teams to use public datasets. This repo uses the organizer-listed **CSE-CIC-IDS2018** CICFlowMeter CSVs from UNB/AWS Open Data.

## Official sources

| Dataset | URL |
|---|---|
| CSE-CIC-IDS2018 (chosen for timestamps + infiltration day) | https://www.unb.ca/cic/datasets/ids-2018.html |
| AWS Open Data bucket | `s3://cse-cic-ids2018/` |
| Direct HTTPS (infiltration day, ~209 MB) | `https://cse-cic-ids2018.s3.ca-central-1.amazonaws.com/Processed Traffic Data for ML Algorithms/Wednesday-28-02-2018_TrafficForML_CICFlowMeter.csv` |
| CIC-IDS2017 | https://www.unb.ca/cic/datasets/ids-2017.html |
| MITRE ATT&CK | https://attack.mitre.org/ |

Cite: Iman Sharafaldin, Arash Habibi Lashkari, Ali A. Ghorbani, “Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization”, ICISSP 2018.

## Commands

From `Normnative-/`:

```bash
pip install pandas

# 1) Download infiltration-day flows (default, most relevant to SIH)
python -m src.world_model.download_cic --dataset infiltration --out data/raw

# 2) Build 5-second network states S_t and S_{t+1} pairs
python -m src.world_model.prepare_dataset --csv data/raw/Wednesday-28-02-2018_TrafficForML_CICFlowMeter.csv --out data/processed/state_windows.npz

# Offline smoke (no download):
python -m src.world_model.prepare_dataset --demo --out data/processed/demo_windows.npz
python tests/test_world_model_windows.py

# 3) Train LSTM world model vs logistic regression (CPU, ~1 min)
python -m src.world_model.train --npz data/processed/state_windows.npz
python tests/test_world_model_train.py
```

`python -m src.world_model.download_cic --list` prints every attack-day file.

## What `S_t` is

- **Δt = 5 seconds**, stride 5s, lookahead **K = 6** (30 seconds).
- 32-d vector: flow-level aggregates (ports, flags, bytes, IAT) + packet-level CICFlowMeter stats (pkt len, TCP window, PSH, port-scan score).
- `ttl_variance`, `ip_fragment_flags`, `retransmit_count` are **0 until PCAP parsing** (not present in CIC ML CSVs).
- Labels: `attack_now`, `attack_within_k`, `infiltration_within_k`, MITRE stage from the CIC attack name.

ML CSVs do **not** include Src/Dst IP. The world model still builds a **service graph** G_t from dest ports in the same 5s bin as S_t (unobserved-client → dest port). A **host graph** (Src IP → Dst IP) is used when those columns exist (GeneratedLabelledFlows / PCAP / the 20 Feb LOIC-HTTP ML CSV). IPs are not stuffed into the frozen 32-d LSTM vector.

## Live dashboard and the 4 GB 20 Feb file

`Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv` (~4 GB, DDoS LOIC-HTTP) is a valid CIC file (Timestamp + Label + Src/Dst IP) but it is **not time-sorted**. Uploading it in the browser will not finish.

Keep files under `data/raw/CIC-IDS2018/` (created automatically). The dashboard lists every `*.csv` under `data/raw/` in a dropdown — CIC-IDS2018 files first — then **RUN LOCAL CSV**. Default live slice is the **latest 90 seconds** (the capture’s “now”). Use **slice: densest attack 90s** only when you want the DDoS/brute peak. Click **refresh list** after copying a new file. Browser uploads stay capped at 80 MB.
