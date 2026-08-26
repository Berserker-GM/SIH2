# Architecture document

**Project:** Predictive Cybersecurity World Model (Normnative)
**Problem:** AI based Network Attack Forecasting from Network Traffic Data
**Length:** 2 pages (print this file)

---

## 1. Purpose

Traditional IDS classifiers label one flow as benign or malicious and throw away time. An infiltration is a **process** (probe → brute → lateral movement). This prototype learns how the **network state** evolves, simulates 30 seconds forward, and tells a defender: something bad? which MITRE stage? which features?

It is a software prototype, fully offline after the public CSE-CIC-IDS2018 CSVs are on disk. There is no cloud inference API.

---

## 2. System overview

```text
  CIC-IDS2018 CSV / local file / synthetic sample
                    │
                    ▼
         Feature pipeline (Python)
         5-second bins, no day glue
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
     S_t ∈ R^32            G_t = (V,E)
     flow + packet         service or host
          │
          ▼
     LSTM world model (frozen)
     input: 8 × 32  (40 s history)
     Head A: Ŝ_{t+1}
     Head B: P(attack within 30 s)
          │
          ▼
     Closed-loop rollout K = 6
     Ŝ_{t+k} is fed back as the next input
          │
     ┌────┴────┬──────────────┐
     ▼         ▼              ▼
  attack P   MITRE stage    why
  (thr 0.15) LogReg on Ŝ    saliency + coef×Ŝ + (Ŝ−S_t)
     │
     ▼
  FastAPI :8001  →  React dashboard :3000
```

**Services**

| Layer | Tech | Role |
|---|---|---|
| Training / eval | PyTorch, NumPy, Pandas, scikit-learn | windows, LSTM, LogReg, metrics |
| Inference | FastAPI + Uvicorn on port **8001** | upload, local CSV, replay, forecast JSON |
| UI | React 19, Vite, Tailwind, Axios | timeline, P(attack), stage, graph, why |
| Optional leftover | Redis, old SOC API :8000, GNN incidents | **not** the world-model result |

No Redis, Docker, or GPU is required to run the SIH demo.

---

## 3. State representation

Each 5-second bin of CICFlowMeter rows becomes:

1. **Vector \(S_t\) (32-d)** — what the LSTM sees.
   - **0–21 flow-level:** counts, unique dest ports, port entropy, TCP/UDP mix, bytes/pkts fwd+bwd, duration, IAT mean/std/max, SYN ACK FIN RST PSH URG, down/up ratio.
   - **22–28 packet-from-CIC:** packet length mean/std/max, init TCP windows, fwd PSH, port-scan score.
   - **29–31 PCAP stubs:** TTL variance, IP fragment flags, retransmit count. Filled only by the Scapy sidecar; frozen LSTM still sees 0 here.
2. **Graph \(G_t\)** — same bin, not LSTM input.
   - CIC ML CSVs omit IPs → **service graph** (unobserved sources → dest ports).
   - When Src/Dst IP exist → **host graph**.

Pairs \((S_t, S_{t+1})\) are kept only if the next bin is within 15 s. Multi-day training concatenates **pairs**, never raw days, so midnight gaps cannot enter a sequence.

---

## 4. World model

| Item | Choice |
|---|---|
| Sequence model | 2-layer LSTM, hidden 64, dropout 0.2 |
| History | \(L=8\) windows = 40 s |
| Loss | MSE(\(\hat{S}_{t+1}, S_{t+1}\)) + 0.5 × weighted BCE(attack-within-K) |
| Optimiser | Adam, lr \(10^{-3}\), weight decay \(10^{-5}\), grad clip 1.0 |
| Split | **Calendar days** (train / val / test), not a shuffled 70/15/15 |
| Freeze | After training: `world_lstm.pt` + train-only `scaler.npz` |

**Forward simulation (the deliverable):**

\[
\hat{S}_{t+1} = f_\theta(S_{t-7:t}), \quad
x \leftarrow [S_{t-6:t},\; \hat{S}_{t+1}], \quad
\text{repeat to } K=6.
\]

Baselines required by the problem statement:

- **Persist:** \(\hat{S}_{t+k} = S_t\) (dynamics).
- **Logistic regression** on the last 32-d window (classifier, no sequence).

MITRE stages are **not** a third LSTM head. A multinomial LogReg is fit on train-day scaled \(S_t\) and applied to \(\hat{S}_{t+6}\). Stages: reconnaissance, initial access, lateral movement, command-and-control, exfiltration (reserved), impact (DoS/DDoS), plus benign.

**Explainability (per prediction):**

- `why_attack` — mean \(|\partial \text{attack logit} / \partial x|\) over the 8-step history.
- `why_stage` — LogReg coefficient of the predicted stage \(\times \hat{S}_{t+6}\).
- `why_change` — \(\hat{S}_{t+6} - S_t\).

---

## 5. Inference path

```text
POST /api/world-model/upload     CIC CSV (≤ 80 MB) → forecast JSON
POST /api/world-model/local      file already in data/raw/
POST /api/world-model/sample     synthetic CIC-shaped rows
GET  /api/world-model/forecast   saved 01 Mar replay
GET  /api/world-model/status     frozen=true, k=6, dim=32
```

Dashboard (`frontend/`, Vite proxy `/api` → `:8001`): attack probability vs 0.15, MITRE stage + technique, 30 s state timeline, \(G_t\) graph, top driving features.

---

## 6. Data and generalisation

Corpus: CSE-CIC-IDS2018 TrafficForML CSVs (UNB / AWS Open Data). Not CTU-13. Not NCIIPC-hosted.

Train days: 14, 15, 16, 22, 28 Feb. Val: 23 Feb. Test: 21 Feb (HOIC), **01 Mar infiltration**, 02 Mar bot.

Quote **01 Mar** for the SIH claim. Do not quote pooled test MSE (21 Feb HOIC, \(z \approx 489\) on packet counts). C2/exfil are unseen in train labels — an admitted coverage gap.

---

## 7. What is not this architecture

The repo still contains an older SOC demo (Redis log simulator, LSTM autoencoder, Isolation Forest, PyG GNN incidents on port 8000). That path is **not** the world model. The SIH scientific claim is the frozen LSTM on `:8001` and the React forecast dashboard.
