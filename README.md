# Predictive Cybersecurity World Model

**Smart India Hackathon** · Problem: *AI based Network Attack Forecasting from Network Traffic Data*

Prototype name: **Normnative** (Normative)

We turn CIC-IDS2018 traffic into a 5-second network state \(S_t\), train an LSTM to learn \(P(S_{t+1} \mid \text{history})\), roll that forecast **30 seconds** ahead, map it to a MITRE ATT&CK stage, and show **why** — in an offline dashboard that accepts a CSV.

This is a **world model**, not a per-flow benign/malicious classifier.

```
CIC CSV  →  32-d S_t every 5s  →  LSTM (8-step memory)
         →  30s closed-loop rollout Ŝ_{t+1} … Ŝ_{t+6}
         →  P(attack in 30s)  +  MITRE stage  +  driving features
         →  React dashboard
```

---

## Quick start (offline demo)

**Requirements:** Python 3.11+, Node.js 18+, ~2 GB RAM. No cloud APIs. No Redis for the world-model path.

```bash
cd Normnative-          # skip this line if this folder is already the repo root
python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

# Terminal A — frozen LSTM API
python -m src.world_model.api          # http://localhost:8001

# Terminal B — dashboard
cd frontend
npm install --legacy-peer-deps
npm run dev                            # http://localhost:3000
```

Open **http://localhost:3000/**

| Action | What it does |
|---|---|
| **PREV / NEXT** | Replay held-out 01 Mar infiltration windows (no CSV needed) |
| **RUN SYNTHETIC SAMPLE** | Same live path as upload, synthetic CIC-shaped rows |
| **UPLOAD CSV** | CIC TrafficForML CSV (`Timestamp` + `Label`, ≤ 80 MB) |
| **RUN LOCAL CSV** | Files already on disk under `data/raw/` (use this for large days) |

The world-model API does **not** need Redis. Redis / the old SOC stack is leftover and is not the SIH claim.

---

## Results to quote (held-out 01 Mar infiltration)

Day-split evaluation. Do **not** quote pooled test MSE (dominated by 21 Feb HOIC scale shift).

| Claim | LSTM | Baseline | Notes |
|---|---:|---:|---|
| Dynamics MSE, 30s rollout | **0.647** | persist 1.105 | ~41% lower error |
| Attack-within-K F1 | **0.602** | LogReg 0.445 | LSTM FPR 0.117 vs 0.914 |
| MITRE stage accuracy on \(\hat{S}\) | **0.800** | persist 0.712 | T1021 lateral movement |
| Lateral-movement F1 | **0.698** | persist 0.584 | same MITRE stage, new day |

Thresholds (chosen on val F1, then frozen): LSTM attack head **0.15**, LogReg **0.05**.

---

## Architecture

```text
[CIC-IDS2018 CSV]
        │
        ▼
[windows.py]  5s aggregate → 32-d S_t  +  graph G_t
        │
        ├── train: calendar-day split, train-only scaler, LSTM
        ▼
[world_lstm.pt + scaler.npz]     FROZEN
        │
        ├── closed-loop rollout K=6 (30 seconds)
        ├── attack head  P(attack in 30s)
        ├── LogReg MITRE decoder (train days only)
        └── saliency / stage contributions / Ŝ−S_t deltas
        ▼
[FastAPI :8001]  ────  [React ForecastDashboard :3000]
```

| Piece | Detail |
|---|---|
| State \(S_t\) | 32-d vector every 5 s (flow 0–21, packet-from-CIC 22–28, PCAP stubs 29–31) |
| Graph \(G_t\) | Same window: service graph on CIC ML CSVs; host graph when Src/Dst IP exist |
| Model | 2-layer LSTM, hidden 64, dropout 0.2, seq_len 8 (40 s of history) |
| Heads | next-state \(\hat{S}_{t+1} \in \mathbb{R}^{32}\) + P(attack in next 6 windows) |
| Rollout | feed \(\hat{S}\) back in, six times — that **is** the world model |
| Stage | multinomial LogReg on scaled 32-d, applied to \(\hat{S}_{t+6}\) |
| Why | attack-head saliency + LogReg feature contributions + forecast deltas |

Two-page architecture write-up: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md). Full report: [`docs/OVERALL_REPORT.md`](docs/OVERALL_REPORT.md).

---

## Dataset

NCIIPC does not host a traffic dump. This prototype uses **CSE-CIC-IDS2018** CICFlowMeter ML CSVs.

- UNB: https://www.unb.ca/cic/datasets/ids-2018.html
- AWS Open Data: `s3://cse-cic-ids2018/` (no account required)
- Helper: `python -m src.world_model.download_cic --list`

```bash
python -m src.world_model.download_cic --dataset infiltration --out data/raw
python -m src.world_model.prepare_dataset --csv-dir data/raw --out data/processed/state_windows_multiday.npz
python -m src.world_model.train --npz data/processed/state_windows_multiday.npz --check-data
```

Raw CSVs and window NPZs are gitignored (~GB). Trained weights (`src/world_model/models/world_lstm.pt`, `scaler.npz`) should ship with the repo.

**Day split (no random 70/15/15 leakage)**

| Split | Days | Role |
|---|---|---|
| Train | 14, 15, 16, 22, 28 Feb | dynamics + scaler + stage decoder |
| Val | 23 Feb | early stop + thresholds |
| Test | 21 Feb, 01 Mar, 02 Mar | HOIC scale shift · held-out infiltration · unseen bot/C2 |

Cite: Iman Sharafaldin, Arash Habibi Lashkari, Ali A. Ghorbani, “Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization”, ICISSP 2018.

More download URLs: [`data/README.md`](data/README.md).

---

## Problem-statement checklist

| Required | Status |
|---|---|
| Ingest network telemetry | Yes — CIC-IDS2018 CSVs |
| Flow-level + packet-level features | Yes — 32-d \(S_t\); PCAP TTL/fragment/retransmit are a Scapy sidecar (frozen dims 29–31 stay 0) |
| State as vector **and** graph | Yes — \(S_t\) + \(G_t\) |
| Learn \(P(S_{t+1} \mid \text{history})\) | Yes — LSTM |
| K-step forward simulation | Yes — K=6 = 30 s |
| P(attacker progression) | Yes — attack-within-K head |
| MITRE ATT&CK stage | Yes — 7 stages (incl. Impact for DoS) |
| Explainability | Yes — saliency + linear attribution + deltas (not SHAP / Transformer attention) |
| Offline CSV → timeline UI | Yes — React + FastAPI |
| LogReg + persist baselines | Yes |
| Generalise to unseen days | Yes — calendar-day split |
| Live packet tap | No — offline CSV / replay |
| Transformer / GNN as the world model | No — LSTM is the WM; leftover SOC GNN is not this claim |

---

## Project layout

```
src/world_model/          # SIH world model (the scientific claim)
  cic_schema.py           32-d feature order
  windows.py              5s states + optional G_t
  pcap_features.py        Scapy TTL / fragment / retransmit sidecar
  state_graph.py          communication graph G_t
  labels.py               CIC Label → MITRE stage
  model.py                LSTM 2×64 (do not change after freeze)
  train.py                day split, LogReg baseline
  rollout.py              closed-loop K-step
  mitre_decode.py         train-only stage LogReg
  explain.py              why-attack / why-stage / why-change
  service.py + api.py     CSV upload + replay API (:8001)
  models/                 world_lstm.pt, scaler.npz, metrics JSON
frontend/                 React + Vite dashboard (:3000)
tests/test_world_model_*.py
data/README.md            how to download CIC CSVs

# Leftover SOC demo — not the world-model result
src/ingestor/  src/ml_models/  src/gnn_fusion/  src/dashboard_api/  simulator/
```

---

## Retrain / evaluate (optional)

Weights in this repo are frozen. Re-run only if you change data or the model.

```bash
python -m src.world_model.prepare_dataset --csv-dir data/raw --out data/processed/state_windows_multiday.npz
python -m src.world_model.train --npz data/processed/state_windows_multiday.npz --check-data
python -m src.world_model.eval_kstep
python -m src.world_model.eval_mitre
python -m src.world_model.eval_explain
```

Smoke tests (no download):

```bash
python -m src.world_model.prepare_dataset --demo --out data/processed/demo_windows.npz
python tests/test_world_model_windows.py
python tests/test_world_model_api.py
```

---

## What we admit

- PCAP TTL / fragment / retransmit are extracted as a sidecar; they are **not** in the frozen 32-d LSTM input.
- C2 / exfiltration are not in the train-day labels, so the decoder cannot emit `command_and_control` on 02 Mar bot.
- 21 Feb HOIC is an out-of-scale day. Pooled test MSE is the wrong number to quote.
- Browser upload is capped at 80 MB. Use **RUN LOCAL CSV** for official day files.

---

## License and data

Code in this repository is the team prototype. CSE-CIC-IDS2018 remains UNB/CIC data — download it from the official sources above; we do not redistribute the raw CSVs.
