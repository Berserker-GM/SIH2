# Smart India Hackathon — Overall Project Report

**Title:** Predictive Cybersecurity World Model
**Problem:** AI based Network Attack Forecasting from Network Traffic Data
**Prototype:** Normnative (Normative)
**Date:** 26 August 2026

This is the one-stop explanation of what we built, why it matches the SIH problem, which technologies we used, how the model was trained, and what is still incomplete.

**Plain idea in one sentence:** we turn CIC-IDS2018 traffic into 5-second network “states”, teach an LSTM how those states evolve, roll the forecast 30 seconds ahead, map it to a MITRE ATT&CK stage, and show **why** — in a dashboard that also accepts a CSV.

---

## 1. The problem, in plain words

SIH does **not** want another flow classifier that labels one row as benign / attack.

It wants a **world model**:

```
Watch the network now (S_t)
  → learn how the network usually changes (P(S_{t+1} | history))
  → simulate a few steps into the future
  → say: something bad in 30s?  what MITRE stage?  which features?
```

An infiltration is a **process** over time (probe → brute → lateral move), not one bad packet. The world model must keep that timeline.

---

## 2. One-page answer: did we build what they asked?

**Yes** for the core world-model loop. **Partial** on PCAP-only fields and GNN-as-dynamics.

| Required by problem statement | Our status |
|---|---|
| Ingest network telemetry | **Yes** — CIC-IDS2018 CICFlowMeter CSV |
| Flow-level features (flags, bytes, IAT) | **Yes** — 22 flow stats in \(S_t\) |
| Packet-level features | **Yes** from CIC columns (len, window, PSH, port-scan score) = dims 22–28. PCAP sidecar (TTL / fragment / retransmit) via Scapy; frozen \(S_t\) dims 29–31 stay 0 (no retrain) |
| Represent state as vector or graph | **Yes** — 32-d vector \(S_t\) every 5s **and** per-window graph \(G_t\) (service graph on CIC ML CSVs; host graph when Src/Dst IP exist). Old SOC GNN is still not the WM |
| Learn \(P(S_{t+1} \mid S_t)\) with LSTM/GNN/etc. | **Yes** — 2-layer LSTM, seq_len 8 |
| Supervised dynamics from attack timeline | **Yes** — next window is the label |
| Forward simulation K steps | **Yes** — closed-loop K=6 = 30 seconds |
| Probability of attacker progression | **Yes** — attack-within-K head, threshold 0.15 |
| MITRE ATT&CK stage mapping | **Yes** — 7 stages (incl. Impact for DoS) |
| Explainability (attention / SHAP / equiv.) | **Yes** — attack-head saliency + LogReg feature contributions + \(\hat{S}-S_t\) deltas (not Transformer attention / SHAP lib) |
| Offline demo: CSV in → timeline out | **Yes** — React dashboard + FastAPI |
| LogReg baseline on same features | **Yes** — last 32-d window, no sequence |
| Persist dynamics baseline | **Yes** — \(\hat{S}_{t+k} = S_t\) |
| Generalise to unseen days | **Yes** — day split (not random 70/15/15) |
| Live packet tap / enterprise sensor | **No** — offline CSV / replay |
| CTU-13 / raw PCAP training | **No** — CIC-IDS2018 ML CSVs only |
| Transformer world model | **No** — LSTM is the WM |
| GNN as the world model | **No** — GNN is leftover SOC demo |

Honest score for a jury: the **world model path is implemented end-to-end**. The leftover Normnative SOC (Redis, autoencoder, GNN incidents) is a separate older demo and is **not** the SIH scientific claim.

---

## 3. What the system does (story a defender sees)

```
Live/current traffic (CSV upload or 01 Mar replay)
     ↓
32-d state every 5 seconds
     ↓
LSTM (8-step memory)
     ↓
30-second predicted future  Ŝ_{t+1} … Ŝ_{t+6}
     ↓
Attack probability          “Something bad in 30s?”
+ MITRE stage               “What is likely to happen?”
+ driving features          “Why does the model think that?”
     ↓
Dashboard (WORLD MODEL · 30s FORECAST)
```

---

## 4. Tech stack (every layer)

**Language / runtime**

- Python 3.11+ (training, windows, LSTM, API)
- JavaScript / React 19 (dashboard)

**Machine learning**

| Library | Use |
|---|---|
| PyTorch | LSTM world model, closed-loop rollout, input saliency |
| scikit-learn | Logistic Regression baseline + MITRE stage decoder |
| NumPy / Pandas | windows, scaler, CIC CSV |

**World-model service**

- FastAPI + Uvicorn — `http://localhost:8001` (no Redis, no cloud)
- `python -m src.world_model.api`

**Dashboard**

- React + Vite + Tailwind — `http://localhost:3000`
- Axios calls `/api/world-model/*`
- Vite proxy: `/api` → `:8001`

**Old SOC demo (not the world model; still in the repo)**

FastAPI `:8000`, Redis, log simulator, Isolation Forest / LSTM autoencoder, PyTorch Geometric GNN, incident assembler. Use it as a SOC skin. Do not quote it as the world-model result.

**Data**

- CSE-CIC-IDS2018 (UNB / AWS Open Data) — CICFlowMeter “TrafficForML” CSVs
- UNB page: https://www.unb.ca/cic/datasets/ids-2018.html
- AWS registry: https://registry.opendata.aws/cse-cic-ids2018/
- S3 (no login): `s3://cse-cic-ids2018/` region `ca-central-1`
- See [section 10](#10-where-to-get-csv-files-to-test) for every test-CSV link

**OS / train hardware**

Windows, CPU-only PyTorch (no CUDA used for the reported run). Open source, fully offline after the CSVs are on disk.

---

## 5. Architecture

### 5.1 Data path (one CSV at a time — never glue raw days together)

```
CIC CSV
  → bind column aliases (CIC-IDS2017/2018 header variants)
  → parse real timestamps (refuse fake clocks)
  → drop repeated header rows / Inf
  → 5-second bins (stride 5s)
  → aggregate flows in the bin → 32-d S_t
  → keep pair (S_t, S_{t+1}) only if the next bin is within 15s
  → per-window MITRE stage from CIC Label (worst stage in the bin)
```

Multi-day NPZ = concatenate **pairs**, not raw rows. Day boundary and timestamp gaps cannot enter a sequence.

### 5.2 World model (frozen after training)

| Item | Value |
|---|---|
| Input | \(x = [S_{t-7},\ldots,S_t]\) shape `(8, 32)` |
| LSTM | 2 layers, hidden 64, dropout 0.2 |
| Head A | next state \(\hat{S}_{t+1} \in \mathbb{R}^{32}\) (dynamics) |
| Head B | logit → P(attack in next 6 windows) ≈ 30s |

Closed-loop rollout (the actual “world model” use):

```
Ŝ_{t+1} = HeadA(LSTM(x))
x ← [S_{t-6}, …, S_t, Ŝ_{t+1}]
repeat to K=6
```

### 5.3 Stage + why (no extra LSTM head)

Train-only multinomial LogReg: scaled 32-d → 7 MITRE stages. Applied to \(\hat{S}_{t+6}\) (and to each \(\hat{S}_{t+k}\) on the upload timeline).

| Signal | Meaning |
|---|---|
| `why_attack` | mean \(\lvert \partial\) attack_logit \(\partial x \rvert\) over the 8-step history |
| `why_stage` | LogReg `coef[stage] × Ŝ_{t+6}` |
| `why_change` | \(\hat{S}_{t+6} - S_t\) (what the forecast thinks will move) |

### 5.4 Inference services

```bash
python -m src.world_model.prepare_dataset
python -m src.world_model.train
python -m src.world_model.eval_kstep
python -m src.world_model.eval_mitre
python -m src.world_model.eval_explain
```

Live API:

| Endpoint | Role |
|---|---|
| `POST /api/world-model/upload` | CIC CSV → full forecast JSON |
| `GET /api/world-model/forecast` | 01 Mar replay examples |
| `POST /api/world-model/forecast` | raw 8×32 history (scaled) |
| `POST /api/world-model/local` | CSV already on disk under `data/raw/` |
| `POST /api/world-model/sample` | synthetic CIC-shaped rows |

Dashboard panel: replay PREV/NEXT + UPLOAD CSV + RUN LOCAL CSV.

### 5.5 ASCII architecture

```
[CIC-IDS2018 CSV]
        |
        v
[windows.py]  5s aggregate → 32-d S_t, stage_id, timestamps
        |
        +-- train: day split, train-only scaler, LSTM train.py
        |
        v
[world_lstm.pt + scaler.npz]     FROZEN
        |
        +-- closed-loop rollout K=6
        +-- attack head P(30s)
        +-- LogReg MITRE decoder (train days only)
        +-- saliency / contributions / deltas
        |
        v
[FastAPI :8001] ---- [React ForecastDashboard :3000]
```

A two-page printable version lives in [`ARCHITECTURE.md`](ARCHITECTURE.md).

---

## 6. Machine learning (what was trained, what was not)

### 6.1 What was trained (once)

| Item | Value |
|---|---|
| Model | `LSTMWorldModel` (`src/world_model/model.py`) |
| Weights | `src/world_model/models/world_lstm.pt` |
| Scaler | `src/world_model/models/scaler.npz` (mean/std of **train** states) |
| Optimiser | Adam, lr=1e-3, weight_decay=1e-5 |
| Batch | 64 |
| Loss | MSE(\(\hat{S}_{t+1}\), \(S_{t+1}\)) + 0.5 × weighted BCE(attack_within_k) |
| Early stop | patience 8 on val loss (multi-day run stopped ~epoch 12) |
| Clip | grad max_norm=1.0 |
| Seq len | 8 windows = 40s of history at 5s bins |
| Horizon K | 6 windows = 30s |

Decision thresholds (chosen on val F1, then frozen):

- LSTM attack head **0.15**
- LogReg attack **0.05**

### 6.2 What was fit but is not the LSTM

- Logistic Regression (binary) — SIH baseline on last 32-d window
- Multinomial LogReg (7 stages) — decoder on scaled \(S_t\), train days only

### 6.3 What was never done (on purpose)

- No Transformer, no extra LSTM `stage_head`, no SHAP library
- No clipping / log-transform of 21 Feb HOIC as a “fix”
- No 70/15/15 shuffle on the multi-day experiment
- No retraining after K-step / MITRE / explain / dashboard / CSV upload

### 6.4 Why LSTM beats “just a classifier”

LogReg sees only \(S_t\). The LSTM sees 8 states and is trained to emit \(S_{t+1}\).

Closed-loop MSE vs persist on held-out infiltration (01 Mar):

| Model | MSE through k=6 |
|---|---:|
| LSTM | **0.647** |
| Persist | 1.105 |

~41% lower error. That is the world-model claim: it learned **dynamics**, not a bag of flags.

---

## 7. Data used to train

Corpus: **CSE-CIC-IDS2018** (CICFlowMeter ML CSVs). Not CTU-13. Not NCIIPC-hosted. Download sites and per-day HTTPS links are in [section 10](#10-where-to-get-csv-files-to-test).

Windowing: each official day-file processed **alone**, then pairs concatenated.

NPZ used for the reported multi-day model: `data/processed/state_windows_multiday.npz` — 9 days, 47,883 pairs, 32-d.

### Day split (calendar days, no leakage)

**Train** (dynamics + scaler + stage decoder)

| Day | Attack | MITRE |
|---|---|---|
| 2018-02-14 | FTP/SSH brute | initial_access (T1110) |
| 2018-02-15 | DoS | impact (T1498) |
| 2018-02-16 | DoS | impact |
| 2018-02-22 | web attacks | initial_access |
| 2018-02-28 | infiltration | lateral_movement (T1021) |

**Val** (threshold + early stop): 2018-02-23 web attacks — new day, seen family.

**Test** (honest generalisation)

| Day | Attack | Why it is a test |
|---|---|---|
| 2018-02-21 | DDoS HOIC / LOIC-UDP | impact — huge scale shift |
| 2018-03-01 | infiltration | same MITRE stage as 28 Feb, **new day** |
| 2018-03-02 | bot | command_and_control — **unseen in train** |

Train decoder class counts (27,340 windows):

| Stage | Count |
|---|---:|
| benign | 22,025 |
| initial_access | 2,453 |
| lateral_movement | 1,596 |
| impact | 1,266 |
| reconnaissance | 0 |
| command_and_control | 0 |
| exfiltration | 0 |

CIC has no exfil label; bot day was held out on purpose.

### 32-d \(S_t\)

- **Flow (0–21):** counts, unique dst ports, port entropy, TCP/UDP ratio, bytes/pkts fwd+bwd, bidirectional ratio, duration, IAT mean/std/max, SYN ACK FIN RST PSH URG sums, down/up ratio
- **Packet from CIC (22–28):** pkt len mean/std/max, init fwd/bwd window, fwd PSH flags, `port_scan_score`
- **PCAP stubs (29–31):** `ttl_variance`, `ip_fragment_flags`, `retransmit_count` = 0
- CIC ML CSVs also omit Src IP / Dst IP / Src Port (reserved 0)

Ground truth for dynamics: the **next** 5s window’s 32-d vector.
Ground truth for attack head: any attack in the next K=6 windows.
Ground truth for MITRE: CIC Label → our label table (e.g. Infilteration → lateral_movement T1021; Bot → C2 T1071; HOIC → impact T1498).

---

## 8. Results to quote (and results not to quote)

### Quote these (world-model / generalisation)

**(A) Dynamics, val 23 Feb** (n≈6,643 k-step windows)

LSTM MSE **0.505** · persist 0.859 · ~41% better at every k including 6

**(B) Dynamics, test 01 Mar infiltration** (n≈6,726)

LSTM MSE **0.647** · persist 1.105 · ~41% better through 30s

**(C) Attack-within-K, test 01 Mar** (n=6,741 sequences)

| Model | F1 | FPR |
|---|---:|---:|
| LSTM | **0.602** | **0.117** |
| LogReg | 0.445 | 0.914 |

LSTM wins with far fewer false alarms.

**(D) MITRE decode of rolled \(\hat{S}\), test 01 Mar**

| | LSTM | Persist |
|---|---:|---:|
| Accuracy | **0.800** | 0.712 |
| lateral_movement F1 | **0.698** | 0.584 |

### Do not quote

Pooled test MSE (LSTM 44 vs persist 9.6). That pool is dominated by 21 Feb HOIC (`pkts_fwd_sum` z≈489). LSTM stays in-distribution; persist copies the flood. It is a **scale shift**, not a parser bug.

Also be honest:

- 02 Mar bot / C2: decoder cannot emit `command_and_control` (unseen in train).
- 23 Feb web: binary head FPR is high; stage decode often stays benign.
- Upload demo: last windows of the CSV; not a live SPAN port.

---

## 9. What was implemented (checklist vs problem statement)

### Background / objectives

- [x] Learn evolving network state from telemetry
- [x] Anticipate attacker progression (30s closed-loop)
- [x] Interpretable defender output (stage + why)
- [x] Software prototype, open-source, offline
- [~] Enterprise / CII applicability — same CICFlowMeter-style features an enterprise sensor can emit; no live tap in this demo

### Section 1 — two levels of features

- [x] Flow-level NetFlow-style aggregates
- [x] Packet-level CICFlowMeter stats + port-scan signature
- [~] TTL variance, IP fragment flags, retransmit counts from PCAP (Scapy sidecar; frozen \(S_t\) still 0 at indices 29–31)
- [ ] Src/Dst IP in the state vector (not in most CIC ML CSVs; 20 Feb LOIC-HTTP *does* have IPs — they feed \(G_t\), not the 32-d LSTM)

### Section 2 — world model architecture

- [x] Structured 32-d state vector at time t
- [x] Per-window communication graph \(G_t\) (service or host)
- [x] Sequence LSTM learns \(P(S_{t+1} \mid S_{t-7:t})\)
- [x] Supervised transitions from the capture timeline
- [x] Held-out days (01 Mar infiltration) to test generalisation
- [ ] Transformer / GNN as the dynamics model

### Section 3 — infiltration prediction + ATT&CK

- [x] Time-series P(attack) over K=6 (and per-step on upload)
- [x] MITRE stages: recon, initial access, lateral movement, C2, exfiltration (reserved), plus impact for DoS/DDoS
- [x] Driving features (saliency + linear attribution + forecast deltas)
- [ ] Built-in LSTM attention weights / SHAP package

### Expected solution (indicative)

- [x] Feature pipeline: CIC-IDS2018 CSV → timestamped 32-d matrix
- [x] Trained WM + weights + train script + config in repo
- [x] K-step engine: P(infiltration), MITRE stage, top features
- [x] Explainability per prediction
- [x] Offline UI: CSV upload (and replay) — React, not Streamlit
- [x] Benchmark vs LogReg (F1, precision, recall, FPR) + persist MSE
- [x] PCAP ingest via Scapy (sidecar; not injected into frozen LSTM)
- [ ] Demo video / 5-slide deck (process deliverables — record after push)

### Code map (`Normnative-/`)

| File | Role |
|---|---|
| `src/world_model/cic_schema.py` | 32-d order + PACKET_LEVEL_SPEC (22–31) |
| `src/world_model/windows.py` | 5s states (optional \(G_t\) sidecar) |
| `src/world_model/pcap_features.py` | Scapy TTL / fragment / retransmit sidecar |
| `src/world_model/state_graph.py` | communication graph \(G_t\) |
| `src/world_model/labels.py` | CIC → MITRE |
| `src/world_model/model.py` | LSTM 2×64 (do not change) |
| `src/world_model/train.py` | day split, LogReg baseline |
| `src/world_model/rollout.py` | closed-loop K-step |
| `src/world_model/mitre_decode.py` | train-only stage LogReg |
| `src/world_model/explain.py` | why-attack / why-stage / why-change |
| `src/world_model/service.py` + `api.py` | CSV upload + replay API |
| `frontend/.../ForecastDashboard.jsx` | defender UI |
| `tests/test_world_model_*.py` | windows, train, rollout, MITRE, explain, API+upload, pcap sidecar |

---

## 10. Where to get CSV files to test

NCIIPC (nciipc.gov.in) does **not** host a traffic dump. The problem statement points teams at public corpora. This prototype uses CSE-CIC-IDS2018 CICFlowMeter CSVs (Timestamp + Label, ~80 columns).

### Official websites

- UNB / CIC dataset page (cite this): https://www.unb.ca/cic/datasets/ids-2018.html
- AWS Registry of Open Data (no AWS account needed): https://registry.opendata.aws/cse-cic-ids2018/
- S3 bucket (`ca-central-1`): `s3://cse-cic-ids2018/`
  - Folder you want: **Processed Traffic Data for ML Algorithms/**
  - Raw PCAPs are a separate multi-hundred-GB folder — not required

AWS CLI (whole CSV folder, ~6–7 GB, not the 450 GB raw capture):

```bash
aws s3 ls --no-sign-request s3://cse-cic-ids2018/
aws s3 sync --no-sign-request \
  "s3://cse-cic-ids2018/Processed Traffic Data for ML Algorithms/" \
  data/raw/CIC-IDS2018/
```

Direct HTTPS prefix (browser or wget; spaces encoded):

`https://cse-cic-ids2018.s3.ca-central-1.amazonaws.com/Processed%20Traffic%20Data%20for%20ML%20Algorithms/`

Repo helper (same HTTPS files, one day at a time):

```bash
cd Normnative-/
python -m src.world_model.download_cic --list
python -m src.world_model.download_cic --dataset infiltration --out data/raw
```

### Day files on that HTTPS prefix

| File | Content |
|---|---|
| `Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv` | FTP/SSH brute |
| `Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv` | DoS GoldenEye / Slowloris |
| `Friday-16-02-2018_TrafficForML_CICFlowMeter.csv` | DoS Hulk / SlowHTTP |
| `Tuesday-20-02-2018_TrafficForML_CICFlowMeter.csv` | DDoS LOIC-HTTP (~4 GB). AWS also ships the official typo name `Thuesday-20-02-2018_...` |
| `Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv` | DDoS HOIC |
| `Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv` | web attacks |
| `Friday-23-02-2018_TrafficForML_CICFlowMeter.csv` | web attacks |
| `Wednesday-28-02-2018_TrafficForML_CICFlowMeter.csv` | infiltration (~209 MB) |
| `Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv` | infiltration (held-out) |
| `Friday-02-03-2018_TrafficForML_CICFlowMeter.csv` | botnet |

Example one-file URL (infiltration day, good first download):

https://cse-cic-ids2018.s3.ca-central-1.amazonaws.com/Processed%20Traffic%20Data%20for%20ML%20Algorithms/Wednesday-28-02-2018_TrafficForML_CICFlowMeter.csv

### How to test this project with those CSVs

| Mode | Notes |
|---|---|
| No download | Dashboard → RUN SYNTHETIC SAMPLE, or PREV/NEXT (saved 01 Mar replay JSON already in the repo) |
| Browser UPLOAD CSV | Needs Timestamp + Label. Cap is 80 MB. Most official day files are larger — slice a few minutes or use local path. Do not upload the 4 GB 20 Feb file. |
| Large files on disk | Put them under `Normnative-/data/raw/` (or `data/raw/CIC-IDS2018/`). Pick from the dropdown → RUN LOCAL CSV |

Optional / related (schema aliases exist; not what we trained on): CIC-IDS2017 at https://www.unb.ca/cic/datasets/ids-2017.html. CTU-13 is listed in the problem statement but is **not** ingested here (different format than CIC TrafficForML CSVs).

**Cite if you publish results:** Iman Sharafaldin, Arash Habibi Lashkari, Ali A. Ghorbani, “Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization”, ICISSP 2018. Plus a link to https://www.unb.ca/cic/datasets/ids-2018.html

---

## 11. How to run (offline)

From `Normnative-/`:

```bash
# data already prepared (when weights are in the repo):
#   src/world_model/models/world_lstm.pt
#   src/world_model/models/scaler.npz

python -m src.world_model.api          # :8001  loads frozen LSTM

cd frontend
npm install --legacy-peer-deps
npm run dev                            # :3000
```

Open http://localhost:3000/

- WORLD MODEL panel → PREV/NEXT (01 Mar replay, no CSV needed)
- RUN SYNTHETIC SAMPLE (no download)
- UPLOAD CSV (CIC TrafficForML, Timestamp + Label, ≤80 MB)
- local CSV dropdown (files in `data/raw` / `CIC-IDS2018`) → RUN LOCAL CSV

Re-train only if asked:

```bash
python -m src.world_model.prepare_dataset --csv-dir data/raw --out data/processed/state_windows_multiday.npz
python -m src.world_model.train --npz data/processed/state_windows_multiday.npz --check-data
```

---

## 12. Jury / slide talking points (keep it simple)

1. We do not classify one flow. We forecast the next network **state**.
2. State = 32 numbers every 5 seconds (flags, ports, bytes, IAT, packet stats) plus a communication graph \(G_t\) for the same bin (dest ports, or hosts if IPs).
3. LSTM looks at 8 states (40s) and outputs \(\hat{S}_{t+1}\) plus P(attack in 30s).
4. We feed \(\hat{S}\) back in, six times — that is the world-model rollout.
5. Persistence (copy \(S_t\)) is the dynamics baseline; LogReg is the classifier baseline. LSTM beats persist by ~40% MSE on held-out infiltration.
6. \(\hat{S}_{t+6}\) is decoded to MITRE (T1021 lateral movement on 01 Mar).
7. Why = which features the attack head, the stage decoder, and the 30s delta care about. Not a black box.
8. Gaps we admit: PCAP TTL/fragment/retransmit are a sidecar (not in frozen \(S_t\)); C2/exfil not in train labels; HOIC day is out of scale. We do not hide those.

---

## 13. Step reports (detail)

Same folder as this report (SIH submission root), and copied under `docs/` when this file lives in the code repo:

- `LSTM_World_Model_Training_Report.txt`
- `LSTM_World_Model_KStep_Report.txt`
- `LSTM_World_Model_MITRE_Report.txt`
- `LSTM_World_Model_Explain_Report.txt`
- `LSTM_World_Model_Dashboard_Report.txt`
- `LSTM_World_Model_State_Graph_Report.txt`
- `LSTM_World_Model_Live_Dashboard_Component_Report.txt`

This file is the overall map. Those files are the lab notebooks.

---

*End of overall report*
