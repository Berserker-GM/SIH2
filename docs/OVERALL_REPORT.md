# Smart India Hackathon — Overall Project Report

Code-repo copy of the submission report. The same document lives at the SIH folder root as `SIH_World_Model_Overall_Report.md`.

**Title:** Predictive Cybersecurity World Model  
**Problem:** AI based Network Attack Forecasting from Network Traffic Data  
**Prototype:** Normnative (Normative)  
**Date:** 26 August 2026

**Plain idea:** we turn CIC-IDS2018 traffic into 5-second network states \(S_t\), teach an LSTM how those states evolve, roll the forecast 30 seconds ahead, map it to a MITRE ATT&CK stage, and show **why** in an offline CSV dashboard.

Two-page architecture: [ARCHITECTURE.md](ARCHITECTURE.md). Setup: [../README.md](../README.md).

---

## 1. The problem

SIH wants a **world model**, not a per-flow classifier.

```
Watch the network now (S_t)
  → learn P(S_{t+1} | history)
  → simulate K steps
  → something bad in 30s?  what MITRE stage?  which features?
```

An infiltration is a process over time, not one bad packet.

---

## 2. Did we build what they asked?

**Yes** for the core loop. **Partial** on PCAP-only fields and GNN-as-dynamics.

| Required | Status |
|---|---|
| Ingest telemetry | Yes — CIC-IDS2018 CICFlowMeter CSV |
| Flow-level features | Yes — 22 flow stats in \(S_t\) |
| Packet-level features | Yes from CIC (dims 22–28). Scapy sidecar for TTL/fragment/retransmit; frozen dims 29–31 stay 0 |
| State as vector **and** graph | Yes — 32-d \(S_t\) + \(G_t\) |
| Learn \(P(S_{t+1} \mid history)\) | Yes — 2-layer LSTM, seq_len 8 |
| K-step simulation | Yes — K=6 = 30 s |
| P(attacker progression) | Yes — attack-within-K, threshold 0.15 |
| MITRE stage | Yes — 7 stages including Impact |
| Explainability | Yes — saliency + LogReg contributions + \(\hat{S}-S_t\) (not SHAP / Transformer attention) |
| Offline CSV → UI | Yes — React + FastAPI |
| LogReg + persist baselines | Yes |
| Day-split generalisation | Yes |
| Live packet tap | No |
| Transformer / GNN as WM | No — LSTM is the WM; leftover SOC GNN is not this claim |

The leftover Redis / autoencoder / GNN SOC is **not** the SIH scientific claim.

---

## 3. Defender story

CSV or 01 Mar replay → 32-d state every 5s → LSTM (8-step memory) → 30s \(\hat{S}_{t+1}\ldots\hat{S}_{t+6}\) → P(attack) + MITRE stage + driving features → dashboard.

---

## 4. Tech stack

| Layer | Tech |
|---|---|
| Training / eval | Python 3.11+, PyTorch, scikit-learn, NumPy, Pandas |
| API | FastAPI + Uvicorn `:8001` — `python -m src.world_model.api` |
| UI | React 19, Vite, Tailwind, Axios `:3000` |
| Data | CSE-CIC-IDS2018 TrafficForML CSVs (UNB / AWS Open Data) |

No Redis and no cloud APIs on the world-model path. Train hardware for the reported run: Windows, CPU-only PyTorch.

Cite: Iman Sharafaldin, Arash Habibi Lashkari, Ali A. Ghorbani, ICISSP 2018. https://www.unb.ca/cic/datasets/ids-2018.html

---

## 5. Architecture

One CSV at a time → bind CIC header aliases → real timestamps → 5s bins → 32-d \(S_t\) → keep \((S_t, S_{t+1})\) only if the next bin is within 15s. Multi-day NPZ concatenates **pairs**, not raw days.

| Item | Value |
|---|---|
| Input | \([S_{t-7},\ldots,S_t]\) shape `(8, 32)` |
| LSTM | 2×64, dropout 0.2 |
| Head A | \(\hat{S}_{t+1} \in \mathbb{R}^{32}\) |
| Head B | P(attack in next 6 windows) |
| Rollout | feed \(\hat{S}\) back, K=6 |
| Stage | train-only multinomial LogReg on scaled 32-d, applied to \(\hat{S}_{t+6}\) |
| Why | `why_attack` saliency, `why_stage` coef×\(\hat{S}\), `why_change` \(\hat{S}_{t+6}-S_t\) |

```
[CIC CSV] → [windows.py] → [world_lstm.pt + scaler.npz]
        → rollout K=6 + attack P + MITRE + why
        → [FastAPI :8001] → [React :3000]
```

---

## 6. Training

| Item | Value |
|---|---|
| Weights | `src/world_model/models/world_lstm.pt` |
| Scaler | train-only `scaler.npz` |
| Loss | MSE(next state) + 0.5 × weighted BCE(attack-within-K) |
| Optimiser | Adam 1e-3, weight decay 1e-5, grad clip 1.0 |
| Early stop | patience 8 (~epoch 12 on the reported run) |
| Thresholds | LSTM 0.15, LogReg 0.05 (val F1, then frozen) |

Not done on purpose: no Transformer, no SHAP, no 70/15/15 shuffle, no HOIC clipping as a “fix”, no retrain after eval/dashboard.

---

## 7. Data

`data/processed/state_windows_multiday.npz` — 9 days, 47,883 pairs, 32-d.

| Split | Days |
|---|---|
| Train | 14, 15, 16, 22, 28 Feb |
| Val | 23 Feb |
| Test | 21 Feb HOIC, **01 Mar infiltration**, 02 Mar bot |

Train decoder counts (27,340 windows): benign 22,025 · initial_access 2,453 · lateral_movement 1,596 · impact 1,266 · recon/C2/exfil **0**.

---

## 8. Results to quote (01 Mar infiltration)

| Claim | LSTM | Baseline |
|---|---:|---:|
| Dynamics MSE (k=6) | **0.647** | persist 1.105 |
| Val dynamics MSE | **0.505** | persist 0.859 |
| Attack-within-K F1 | **0.602** | LogReg 0.445 |
| Attack-within-K FPR | **0.117** | LogReg 0.914 |
| MITRE acc on \(\hat{S}\) | **0.800** | persist 0.712 |
| Lateral-movement F1 | **0.698** | persist 0.584 |

**Do not quote** pooled test MSE (LSTM 44 vs persist 9.6) — 21 Feb HOIC scale shift.

Admitted gaps: 02 Mar C2 unseen in train; 23 Feb web FPR high; demo is offline CSV, not a SPAN port.

---

## 9. Checklist vs problem statement

Done: evolving state, 30s rollout, interpretable stage+why, open-source offline prototype, flow+packet CIC features, \(S_t\)+\(G_t\), LSTM dynamics, day-split generalisation, MITRE mapping, saliency explainability, CSV UI, LogReg+persist benchmarks, Scapy PCAP sidecar.

Partial: PCAP dims 29–31 not in frozen LSTM; enterprise live tap; Src/Dst IP not in most CIC ML CSVs.

Not this claim: Transformer/GNN as the world model; SHAP package; demo video / 5-slide deck (record after GitHub push).

---

## 10–12. Data URLs, run, talking points

Download CIC CSVs from UNB / `s3://cse-cic-ids2018/`. Repo helper:

```bash
python -m src.world_model.download_cic --dataset infiltration --out data/raw
python -m src.world_model.api
```

Dashboard: PREV/NEXT (no CSV), RUN SYNTHETIC SAMPLE, UPLOAD CSV (≤80 MB), RUN LOCAL CSV for large days.

Talking points: forecast **state**, not a flow label; 32-d every 5s plus \(G_t\); 8-step LSTM; closed-loop K=6; persist and LogReg baselines; T1021 on 01 Mar; admit sidecar PCAP, missing C2/exfil labels, and HOIC scale.

Lab notebooks (SIH folder): `LSTM_World_Model_*.txt`.
