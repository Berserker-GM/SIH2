"""Why-explanations for the frozen LSTM world model.

Answers three SIH questions without changing architecture or weights:
  - Something bad?     attack head on current history (threshold 0.15)
  - What will happen?  MITRE stage decoded from Ŝ_{t+K}
  - Why?               attack saliency + LogReg stage contributions + Ŝ−S_t deltas
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn

from src.world_model.cic_schema import STATE_FEATURE_ORDER
from src.world_model.labels import STAGE_NAMES, technique_for_stage
from src.world_model.mitre_decode import StageDecoder
from src.world_model.rollout import DEFAULT_K, rollout_closed_loop

LSTM_ATTACK_THRESHOLD = 0.15
DEFAULT_TOP_K = 5


def attack_saliency(model: nn.Module, x: torch.Tensor | np.ndarray) -> np.ndarray:
    """Mean |∂ attack_logit / ∂x| over the history length. x: (n, seq, dim)."""
    if isinstance(x, np.ndarray):
        xt = torch.from_numpy(np.asarray(x, dtype=np.float32))
    else:
        xt = x
    if xt.ndim != 3:
        raise ValueError(f"x must be (n, seq, dim), got {tuple(xt.shape)}")
    n, _, dim = xt.shape
    model.eval()
    xt = xt.detach().clone().requires_grad_(True)
    _, logit = model(xt)
    if not torch.is_tensor(logit) or not logit.requires_grad:
        return np.zeros((n, dim), dtype=np.float32)
    logit.sum().backward()
    grad = xt.grad
    if grad is None:
        return np.zeros((n, dim), dtype=np.float32)
    return grad.abs().mean(dim=1).detach().cpu().numpy().astype(np.float32)


def _coef_for_stage(decoder: StageDecoder, stage_id: int) -> np.ndarray:
    clf = decoder.clf
    n_feat = int(clf.coef_.shape[1])
    classes = [int(c) for c in clf.classes_]
    if stage_id not in classes:
        return np.zeros(n_feat, dtype=np.float64)
    if clf.coef_.shape[0] == 1:
        pos = classes[-1]
        coef = clf.coef_[0].astype(np.float64)
        return coef if stage_id == pos else -coef
    return clf.coef_[classes.index(stage_id)].astype(np.float64)


def stage_contributions(
    decoder: StageDecoder,
    state: np.ndarray,
    stage_id: int,
) -> np.ndarray:
    """Linear LogReg feature contributions for one stage: coef * x."""
    x = np.asarray(state, dtype=np.float64).reshape(-1)
    coef = _coef_for_stage(decoder, int(stage_id))
    if coef.shape[0] != x.shape[0]:
        raise ValueError(f"coef dim {coef.shape[0]} != state dim {x.shape[0]}")
    return (coef * x).astype(np.float64)


def top_features(
    scores: np.ndarray,
    names: list[str] | tuple[str, ...] = STATE_FEATURE_ORDER,
    k: int = DEFAULT_TOP_K,
) -> list[dict[str, Any]]:
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if len(scores) != len(names):
        raise ValueError("scores length must match feature names")
    order = np.argsort(-np.abs(scores))
    out = []
    for i in order[: int(k)]:
        s = float(scores[i])
        out.append({
            "name": str(names[i]),
            "index": int(i),
            "score": s,
            "abs_score": abs(s),
            "direction": "up" if s >= 0 else "down",
        })
    return out


def _narrate(
    *,
    something_bad: bool,
    attack_probability: float,
    attack_threshold: float,
    stage: str,
    technique_id: str,
    technique_name: str,
    why_attack: list[dict],
    why_stage: list[dict],
    why_change: list[dict],
) -> str:
    feat = why_attack[0]["name"] if why_attack else "input features"
    change = why_change[0]["name"] if why_change else "state"
    stage_feat = why_stage[0]["name"] if why_stage else "state features"
    tech = f"{technique_id} {technique_name}".strip() if technique_id else "none"
    if something_bad:
        head = (
            f"Something bad is likely in the next 30s "
            f"(attack probability {attack_probability:.2f} ≥ {attack_threshold:.2f})."
        )
    else:
        head = (
            f"No attack alert "
            f"(probability {attack_probability:.2f} < {attack_threshold:.2f})."
        )
    what = f" What is likely: {stage}" + (f" ({tech})." if tech != "none" else ".")
    why = (
        f" Why: the attack head is driven most by {feat}; "
        f"the stage decode leans on {stage_feat}; "
        f"the 30s forecast changes {change} most."
    )
    return head + what + why


def explain_forecast(
    model: nn.Module,
    decoder: StageDecoder,
    hist: np.ndarray,
    k: int = DEFAULT_K,
    attack_threshold: float = LSTM_ATTACK_THRESHOLD,
    top_k: int = DEFAULT_TOP_K,
) -> dict[str, Any]:
    """
    hist: (seq, dim) or (1, seq, dim) scaled states ending at S_t.
    Returns dashboard-ready dict. Does not train.
    """
    arr = np.asarray(hist, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[None, ...]
    if arr.ndim != 3:
        raise ValueError(f"hist must be (seq, dim) or (n, seq, dim), got {arr.shape}")
    if arr.shape[0] != 1:
        raise ValueError("explain_forecast explains one history at a time")

    states, logits = rollout_closed_loop(model, arr, k=k)
    model.eval()
    with torch.no_grad():
        _, logit0 = model(torch.from_numpy(arr))
        attack_probability = float(torch.sigmoid(logit0[0]).cpu())
    attack_probs_k = torch.sigmoid(torch.from_numpy(logits[0])).numpy().astype(np.float64)

    s_now = arr[0, -1]
    s_hat = states[0, -1]
    stage_id = int(decoder.predict(s_hat[None, :])[0])
    stage = STAGE_NAMES[stage_id]
    tid, tname = technique_for_stage(stage)
    something_bad = attack_probability >= float(attack_threshold)

    why_attack = top_features(attack_saliency(model, arr)[0], STATE_FEATURE_ORDER, k=top_k)
    why_stage = top_features(stage_contributions(decoder, s_hat, stage_id), STATE_FEATURE_ORDER, k=top_k)
    why_change = top_features(s_hat - s_now, STATE_FEATURE_ORDER, k=top_k)

    narrative = _narrate(
        something_bad=something_bad,
        attack_probability=attack_probability,
        attack_threshold=float(attack_threshold),
        stage=stage,
        technique_id=tid,
        technique_name=tname,
        why_attack=why_attack,
        why_stage=why_stage,
        why_change=why_change,
    )
    return {
        "attack_probability": attack_probability,
        "attack_threshold": float(attack_threshold),
        "something_bad": bool(something_bad),
        "stage": stage,
        "stage_id": stage_id,
        "technique_id": tid,
        "technique_name": tname,
        "forecast_states": states[0].astype(np.float32),
        "attack_probs_k": attack_probs_k,
        "why_attack": why_attack,
        "why_stage": why_stage,
        "why_change": why_change,
        "narrative": narrative,
        "k": int(k),
        "horizon_seconds": int(k) * 5,
    }
