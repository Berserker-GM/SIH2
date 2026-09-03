"""
Deterministic (1+1)-style hill-climbing for adaptive/evasive personas.

No LLM anywhere in this loop. Each cycle:
  1. perturb theta (1-2 dims, small step, clipped to the config's search_space bounds)
  2. regenerate a short batch of traffic under candidate theta
  3. score the most recent 8 windows with score_history()
  4. accept iff attack_probability dropped from the current best AND the
     persona still reached its target kill-chain phase within the step
     budget (hard constraint — going silent is not evasion)
  5. otherwise revert

Logs every cycle to personas/adaptation_logs/<persona_id>/<run_id>.jsonl.

IMPORTANT: `score_history` is imported from generator.score_stub, which is a
local placeholder (see that file's docstring). Swap this import for the
pipeline dev's real in-process function once its signature is confirmed —
that's the only change needed here.
"""
import copy
import json
import uuid
from pathlib import Path

import numpy as np

from .engine import generate_run
from .schema import FEATURE_COLUMNS, SCORE_HISTORY_WINDOWS
from .score_stub import score_history  # PLACEHOLDER — replace with pipeline dev's real function

META_KEYS = {"kill_chain_target_phase", "step_budget_windows"}


def _get_nested(d, path_parts):
    cur = d
    for p in path_parts:
        cur = cur[p]
    return cur


def _set_nested(d, path_parts, value):
    cur = d
    for p in path_parts[:-1]:
        cur = cur.setdefault(p, {})
    cur[path_parts[-1]] = value


def _theta_from_config(cfg, search_space):
    theta = {}
    for key in search_space:
        if key in META_KEYS:
            continue
        phase_name, *rest = key.split(".")
        phase = next(p for p in cfg["phases"] if p["name"] == phase_name)
        theta[key] = _get_nested(phase, rest)
    return theta


def _apply_theta(cfg, theta):
    cfg = copy.deepcopy(cfg)
    for key, value in theta.items():
        phase_name, *rest = key.split(".")
        phase = next(p for p in cfg["phases"] if p["name"] == phase_name)
        _set_nested(phase, rest, value)
    return cfg


def _perturb(theta, search_space, rng, n_dims=None):
    keys = [k for k in theta if k not in META_KEYS]
    n_dims = n_dims or rng.integers(1, 3)  # 1 or 2 dims per cycle, per the brief
    chosen = rng.choice(keys, size=min(n_dims, len(keys)), replace=False)
    new_theta = dict(theta)
    for k in chosen:
        bounds = search_space[k]
        step = bounds["step"]
        delta = float(rng.choice([-1, 1])) * step
        new_val = new_theta[k] + delta
        new_val = max(bounds["min"], min(bounds["max"], new_val))
        new_theta[k] = round(new_val, 6)
    return new_theta


def _score_batch(cfg, run_id, seed_offset, budget_windows, target_phase):
    run_id_out, rows = generate_run(cfg, total_windows=budget_windows, run_id=run_id, seed_offset=seed_offset)
    kill_chain_reached = any(r["_phase_name"] == target_phase for r in rows)
    last_phase = rows[-1]["_phase_name"]

    tail = rows[-SCORE_HISTORY_WINDOWS:]
    if len(tail) < SCORE_HISTORY_WINDOWS:
        tail = [rows[0]] * (SCORE_HISTORY_WINDOWS - len(tail)) + tail
    x = np.array([[float(r[f]) for f in FEATURE_COLUMNS] for r in tail])

    result = score_history(x, model_version="v2")
    return result, kill_chain_reached, last_phase


def run_hill_climb(cfg: dict, cycles: int = 15, log_dir: str = "personas/adaptation_logs"):
    assert cfg["adaptive"], f"{cfg['persona_id']} is not marked adaptive"
    search_space = cfg["search_space"]
    target_phase = search_space["kill_chain_target_phase"]
    budget = int(search_space["step_budget_windows"])

    rng = np.random.default_rng(cfg["base_seed"] + 9999)
    theta = _theta_from_config(cfg, search_space)

    run_id = f"{cfg['persona_id']}__adapt__{uuid.uuid4().hex[:10]}"
    log_path = Path(log_dir) / cfg["persona_id"] / f"{run_id}.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Common random numbers: every cycle evaluates its candidate theta against
    # the SAME underlying seed (seed_offset fixed, not cycle-varying). This
    # isolates the effect of theta from resampling noise — without it, a
    # lucky-noise baseline at cycle 0 can never be beaten regardless of theta.
    EVAL_SEED_OFFSET = 424242

    # cycle 0: baseline
    baseline_cfg = _apply_theta(cfg, theta)
    result, kc_reached, last_phase = _score_batch(baseline_cfg, f"{run_id}_c0", EVAL_SEED_OFFSET, budget, target_phase)
    best_prob = result["attack_probability"]
    best_theta = theta

    log_entries = [{
        "cycle": 0, "theta": theta, "attack_probability": best_prob,
        "kill_chain_phase": last_phase, "kill_chain_reached_target": kc_reached,
        "accepted": True, "note": "baseline",
    }]

    for cycle in range(1, cycles + 1):
        candidate = _perturb(best_theta, search_space, rng)
        cand_cfg = _apply_theta(cfg, candidate)
        result, kc_reached, last_phase = _score_batch(
            cand_cfg, f"{run_id}_c{cycle}", EVAL_SEED_OFFSET, budget, target_phase
        )
        prob = result["attack_probability"]

        accept = kc_reached and (prob < best_prob)
        if accept:
            best_prob = prob
            best_theta = candidate

        log_entries.append({
            "cycle": cycle, "theta": candidate, "attack_probability": prob,
            "kill_chain_phase": last_phase, "kill_chain_reached_target": kc_reached,
            "accepted": accept,
        })

    with open(log_path, "w") as f:
        for entry in log_entries:
            f.write(json.dumps(entry) + "\n")

    return {
        "persona_id": cfg["persona_id"],
        "run_id": run_id,
        "log_path": str(log_path),
        "baseline_attack_probability": log_entries[0]["attack_probability"],
        "final_attack_probability": best_prob,
        "final_theta": best_theta,
        "cycles": cycles,
    }
