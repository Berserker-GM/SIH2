# Synthetic Network Traffic Generator — 20-Agent Persona Suite

Deterministic (no-LLM-in-the-loop) traffic generator producing pre-aggregated
32-feature, 5-second-window rows matching the shared SIH world-model pipeline
schema exactly.

## Quick start

```bash
pip install numpy pandas

# 1. Build all 20 persona configs from scratch (idempotent — overwrites personas/configs/*.json)
python3 build_configs.py

# 2. Generate one run per persona (1 simulated hour = 720 windows by default)
python3 run_all.py --windows 720
# -> data/raw/personas/<persona_id>/<run_id>.csv

# 3. Run the hill-climbing search for the 3 adaptive/evasive personas (pre-demo, offline)
python3 run_adaptive.py --cycles 20
# -> personas/adaptation_logs/<persona_id>/<run_id>.jsonl

# 4. One-time divergence check for dataset-calibrated personas
python3 validate.py
# -> docs/divergence_report.csv

# 5. Consolidated MITRE reasoning doc for recon/C2/exfil personas
python3 generate_mitre_notes.py
# -> docs/mitre_notes.md
```

## Layout

```
generator/
  schema.py       # CSV_COLUMNS, MITRE_STAGES — the shared contract, in one place
  config_io.py     # config loading + validation (fills defaults, no per-persona branching)
  sampler.py       # per-5s-window feature sampler (the 32 columns)
  engine.py        # phase state machine + CSV writer
  score_stub.py     # PLACEHOLDER mock of the pipeline dev's score_history() — see below
  adaptive.py       # (1+1) hill-climbing loop for evasive personas
build_configs.py    # defines & writes all 20 persona JSON configs
run_all.py           # generate CSVs for all (or one) persona
run_adaptive.py       # run the hill-climbing search
validate.py            # divergence check vs reference stats
generate_mitre_notes.py # pulls config _notes into docs/mitre_notes.md
personas/configs/*.json         # the 20 persona configs
personas/reference_stats.json   # PLACEHOLDER per-class stats (see below)
personas/adaptation_logs/       # hill-climbing cycle logs (jsonl)
data/raw/personas/<id>/<run_id>.csv
docs/mitre_notes.md              # consolidated recon/C2/exfil reasoning
docs/divergence_report.csv       # validate.py output
```

## Status against the acceptance criteria

- [x] ~20 persona configs following the shared JSON schema exactly (20 written by `build_configs.py`)
- [x] Generator produces CSVs matching the shared 32-column schema exactly, column order included
      (`generator/schema.py::CSV_COLUMNS` is asserted against on every write; verified against a
      generated file — header equality checked directly)
- [x] Recon/C2/exfil personas documented with MITRE technique ID + reasoning (`docs/mitre_notes.md`,
      also inline as `_notes` in each config)
- [x] 2–3 adaptive personas with hill-climbing cycle logs showing attack-probability-over-cycles,
      kill-chain progression preserved (3 personas, 20 cycles each, hard kill-chain constraint
      enforced and verified on every accepted step — see `personas/adaptation_logs/`)
- [x] One-time divergence check vs reference stats for dataset-calibrated personas (`validate.py`,
      mechanism fully built and tested)
- [ ] **Confirmed `score_history()` signature with the pipeline dev before building the adaptive
      loop against it — NOT YET DONE.** This is a hard dependency called out explicitly in the
      brief and I haven't had that conversation. Everything above runs today against
      `generator/score_stub.py`, a local mock with the *documented* signature
      (`x: shape (8,32) -> {attack_probability, something_bad, stage, technique_id,
      why_attack_top_features}`), clearly marked as a placeholder in its own docstring. The
      **only** change needed once the real function ships is the import line at the top of
      `generator/adaptive.py`. Do not treat `score_stub.py`'s attack-probability numbers as
      meaningful outside this repo — its weights are illustrative, not calibrated against the
      real LSTM.

## Known placeholders that need real numbers before this is demo-final

1. **`personas/reference_stats.json`** — the per-family mean/std used by `validate.py` are
   illustrative placeholders, not pulled from the pipeline dev's actual CIC-IDS2018 training
   report. The brief is explicit that these should come from asking the pipeline dev rather than
   re-deriving them. Swap this file's contents once you have those numbers; `validate.py`'s
   z-score mechanism itself needs no changes. Current run already surfaces one useful finding
   worth knowing about regardless: **`dest_port_entropy` and `pkt_len_mean` vary a lot within the
   `initial_access` family** (raw SSH/FTP brute force vs. web-form brute force vs. SQLi/XSS look
   very different on those two features) — a single family-level reference bucket is probably too
   coarse; you likely want per-CICIDS-subclass reference stats, not one bucket per `mitre_stage`.
2. **`generator/score_stub.py`** — replace with the real `score_history()` import once confirmed
   (see above).

## Design notes

- **Dataset-calibrated vs MITRE-doc-based vs adaptive** are just three different *sources of
  truth for the numeric knobs* — all three tiers share the exact same JSON config shape
  (`timing` / `flags` / `ports` / `payload` / `packet_fields` / `phases`), so `engine.py` and
  `sampler.py` have zero persona-specific branching.
- **Overlap/noise injection**: recon/C2/exfil personas (and the evasive variants) carry a
  `noise.stage_overlap_pct` that blends a fraction of a window's features toward a benign
  reference profile without changing the ground-truth label — this keeps those classes from
  being trivially separable, per the brief's explicit warning about inflated F1 on those classes.
- **`run_id` discipline**: `engine.generate_run()` mints a fresh `run_id` every call and every
  row in a run shares it; nothing merges two runs into one sequence.
- **Adaptive loop uses common random numbers**: each hill-climbing cycle evaluates its candidate
  θ against the same underlying seed rather than a fresh one per cycle. Without this, resampling
  noise between cycles swamped the (real) effect of θ and made the very first baseline draw
  unbeatable regardless of how good later parameters were — worth knowing if this gets
  re-implemented against the real `score_history()`, since a real LSTM scorer will have its own
  variance characteristics that may need the same treatment (or more hill-climbing cycles).
