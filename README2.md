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
- [x] Generator produces CSVs matching the shared 32-column schema exactly, column order included —
      confirmed against the pipeline dev's v2 report: `STATE_FEATURE_ORDER` in `cic_schema.py`
      matches this repo's `CSV_COLUMNS[5:]` with no mismatch to reconcile.
- [x] Recon/C2/exfil personas documented with MITRE technique ID + reasoning (`docs/mitre_notes.md`)
- [x] 2–3 adaptive personas with hill-climbing cycle logs showing attack-probability-over-cycles,
      kill-chain progression preserved (3 personas, 20 cycles each, hard kill-chain constraint
      enforced and verified on every accepted step)
- [x] One-time divergence check vs reference stats for dataset-calibrated personas (`validate.py`)
- [x] `score_history()` signature confirmed — pipeline dev's v2 report ships it in
      `src/world_model/service.py`, real (not HTTP), in-process, matching the documented shape
      plus extra fields (`ood_score`, `ood_flag`, `ood_threshold`, `why_stage`, `why_change`,
      `narrative`). `generator/adaptive.py` now imports the real function first and only falls
      back to the local `score_stub.py` mock if that import fails (i.e. when this repo is run
      standalone, outside the merged pipeline tree — which is the case in this sandbox, hence
      the `[warn]` lines when you run `run_adaptive.py` here). **No further code change needed
      on merge** — once this repo sits inside the pipeline tree, the real import wins
      automatically and every hill-climbing log gets tagged `"_scorer": "real:score_history"`
      instead of `"placeholder:score_stub"`, so old stub-scored logs can never be mistaken for
      real-model results later.

## What's next (post-merge) — priority order

The pipeline dev's v2 report changes what "done" means here. The original blocker
(confirm `score_history()`) is resolved; the report's own findings point at the next
real problem:

1. **Mass, not signature, is what's blocking C2/recon/exfil.** Their v2 was trained on 39
   fixture windows per class for recon/C2/exfiltration. Real 02 Mar C2 recall was 0.000, and —
   more tellingly — recall was *also* 0.000 on held-out **fixture** C2, meaning 39 windows isn't
   enough signal at any distribution, real or synthetic. `produce_dataset.py` (new in this drop)
   fixes this directly: recon/C2/exfil personas now get 8 runs × 200 windows = ~1,600 windows
   each (5,078 recon / 4,700 C2 / 1,600 exfiltration windows in aggregate — see
   `docs/dataset_manifest.json`), landing in the same order of magnitude as the smallest real
   CIC class (impact: 1,305 windows) instead of 39. **This needs an actual retrain + eval cycle
   I can't run from this sandbox** — it depends on their `prepare_dataset.py` / `train.py`
   (section 11 of their report):
   ```
   python -m src.world_model.prepare_dataset --cic-npz data/processed/state_windows_multiday.npz \
     --personas-dir data/raw/personas --write-persona-fixtures \
     --out data/processed/state_windows_combined.npz
   python -m src.world_model.train --npz data/processed/state_windows_combined.npz --tag v2
   python -m src.world_model.eval_v2 --npz-v1 data/processed/state_windows_multiday.npz \
     --npz-v2 data/processed/state_windows_combined.npz
   ```
   Drop `data/raw/personas/` from this repo in place of the fixture directory first.

2. **Confirm `mitre_stage` strings match `STAGE_NAMES` in `labels.py` exactly.** Their report
   says lookup is direct, no CIC Label mapping — so a silent string mismatch (e.g. casing,
   underscores) would fail quietly rather than error loudly. This repo emits exactly:
   `benign, reconnaissance, initial_access, lateral_movement, command_and_control, exfiltration,
   impact` (see `generator/schema.py::MITRE_STAGES`). Worth a direct diff against their
   `labels.py` before the next retrain, not an assumption.

3. **Watch the TTL-shortcut risk they flagged.** Their `why_stage`/`why_change` output already
   cites `ttl_variance` / `ip_fragment_flags` / `retransmit_count` on the evasive_lateral fixture
   — their report explicitly calls this "a generator knob: if her evasive personas only jitter
   TTL, the stage head will key on it." `evasive_lateral_v1`'s hill-climbing search space here
   varies `packet_fields.fragment_rate` alongside `timing.lambda` and
   `flags.port_scan_score_gain` (3 dims, not TTL alone) — but this is exactly the kind of thing
   that's easy to get subtly wrong and only shows up in saliency output after a real retrain.
   Worth re-checking `why_stage` output on the new (non-fixture) evasive personas once v2 is
   retrained on this drop, not assuming the 3-dim search space already fixed it.

4. **Re-run `run_adaptive.py` against the real scorer once merged.** Every result in
   `personas/adaptation_logs/` right now is scored by the local placeholder mock (tagged
   `"_scorer": "placeholder:score_stub"` in every log entry) — its weights are illustrative, not
   the real LSTM's. The loop itself (perturb → regenerate → common-random-numbers rescoring →
   accept/revert with the kill-chain hard constraint) doesn't need to change; only the scorer
   swaps automatically on merge (see above). Numbers will very likely differ once real
   `attack_probability` replaces the mock's logistic — that's expected, re-run and take the new
   numbers, don't average the two.

5. **`personas/reference_stats.json` is still placeholder**, not real CIC-IDS2018 per-class
   stats — unrelated to the v2 report, still open from before. The check already surfaced that a
   single `initial_access` bucket is too coarse (web attacks vs. raw brute force diverge on
   `dest_port_entropy` and `pkt_len_mean`, z-scores 2–3.4) — worth splitting the reference buckets
   before trusting `validate.py`'s flags, once real numbers are available.

## Confirmed compatible, no action needed

- **Schema**: `STATE_FEATURE_ORDER` == this repo's `FEATURE_COLUMNS`, same order, per their report.
- **Run/day discipline**: "each `(persona_id, run_id)` is one synthetic day, sequences never cross
  a run" — matches `engine.generate_run()`'s fresh-`run_id`-per-call design exactly.
- **Dims 29–31 (`ttl_variance`, `ip_fragment_flags`, `retransmit_count`) are real, not stubbed** —
  their report notes v1's scaler had these at std=1.0 (floor, all-zero real data) and v2 now sees
  real variance (0.36 / 0.13 / 0.30) from this generator's `packet_fields` sampling. No change needed.
- **Never used the upload API** — their report says bulk persona data must not go through
  `POST /upload`; this repo only ever writes directly to `data/raw/personas/<persona_id>/<run_id>.csv`.
- **Run-id naming now matches their split convention**: `run_all.py` / `produce_dataset.py` name
  runs `<persona_id>__run_000`, `__run_001`, `__run_002`, ... so their TRAIN/VAL/TEST-by-run_id
  split (run_000/001/002) applies directly without renaming.

## Production dataset (this drop)

Generated by `produce_dataset.py`, sized per the reasoning in item 1 above. Full breakdown in
`docs/dataset_manifest.json`; aggregate windows per `mitre_stage` (synthetic side only):

```
initial_access     10,800
benign               6,480
lateral_movement     5,442
reconnaissance       5,078
command_and_control  4,700
impact               4,320
exfiltration         1,600
```

Regenerate with `python3 produce_dataset.py --clean`.

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
