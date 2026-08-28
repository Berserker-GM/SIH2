"""Closed-loop K-step rollout of the world-model next-state head.

Does not change LSTM architecture. Intended for evaluation of a trained
checkpoint: feed predicted S_{t+1} back in to get S_{t+2} ... S_{t+K}.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn

from src.world_model.dataset import DAY_ID_DTYPE, canonical_day_id
from src.world_model.metrics import next_state_metrics

DEFAULT_K = 6
DEFAULT_SEQ_LEN = 8
DEFAULT_MAX_GAP = 15.0


def persist_rollout(last_state: np.ndarray, k: int) -> np.ndarray:
    """Repeat S_t for K steps. last_state: (n, dim) -> (n, k, dim)."""
    last = np.asarray(last_state, dtype=np.float32)
    if last.ndim != 2:
        raise ValueError(f"last_state must be (n, dim), got {last.shape}")
    return np.repeat(last[:, None, :], int(k), axis=1)


def _model_device(model: nn.Module, fallback: torch.device) -> torch.device:
    params = list(model.parameters())
    return params[0].device if params else fallback


@torch.no_grad()
def rollout_closed_loop(
    model: nn.Module,
    x: torch.Tensor | np.ndarray,
    k: int = DEFAULT_K,
    batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Closed-loop forecast.

    x: (n, seq_len, dim) history ending at S_t
    returns:
      states: (n, k, dim)  predicted S_{t+1} ... S_{t+k}
      logits: (n, k)       attack head at each rollout step
    """
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(np.asarray(x, dtype=np.float32))
    if x.ndim != 3:
        raise ValueError(f"x must be (n, seq_len, dim), got {tuple(x.shape)}")
    n, _, dim = x.shape
    device = _model_device(model, x.device)
    model.eval()

    pred_chunks: list[np.ndarray] = []
    logit_chunks: list[np.ndarray] = []
    for start in range(0, n, batch_size):
        hist = x[start:start + batch_size].to(device)
        step_states = []
        step_logits = []
        for _ in range(int(k)):
            nxt, logit = model(hist)
            step_states.append(nxt)
            step_logits.append(logit)
            hist = torch.cat([hist[:, 1:, :], nxt.unsqueeze(1)], dim=1)
        pred_chunks.append(torch.stack(step_states, dim=1).cpu().numpy())
        logit_chunks.append(torch.stack(step_logits, dim=1).cpu().numpy())

    if not pred_chunks:
        return (
            np.zeros((0, int(k), dim), dtype=np.float32),
            np.zeros((0, int(k)), dtype=np.float32),
        )
    return np.concatenate(pred_chunks, axis=0), np.concatenate(logit_chunks, axis=0)


def collect_kstep_windows(
    states: np.ndarray,
    next_states: np.ndarray,
    timestamps: np.ndarray,
    day_ids: np.ndarray,
    seq_len: int = DEFAULT_SEQ_LEN,
    k: int = DEFAULT_K,
    max_gap_seconds: float = DEFAULT_MAX_GAP,
    stage_ids: np.ndarray | None = None,
) -> dict[str, Any]:
    """
    Histories of length `seq_len` that also have K valid future pairs.

    Future targets are next_states[i : i+k] == S_{t+1} ... S_{t+k}
    only when indices [i-seq_len+1, ..., i+k-1] are same-day and
    adjacent timestamps differ by at most max_gap_seconds.

    If stage_ids is provided, also return y_stage[i, j] = stage of S_{t+j+1}
    which is stage_id[i+1+j] (needs one extra trailing pair: last_i = n-k-1).
    """
    n = len(states)
    days = np.array([canonical_day_id(d) for d in day_ids])
    ts = np.asarray(timestamps, dtype=np.float64)
    n_rejected_day = n_rejected_gap = n_rejected_order = 0
    xs, ys, out_days, out_ts, out_stage = [], [], [], [], []

    stages = None
    if stage_ids is not None:
        stages = np.asarray(stage_ids)
        if len(stages) != n:
            raise ValueError("stage_ids length must match states")
        last_i = n - k - 1
    else:
        last_i = n - k
    for i in range(seq_len - 1, last_i + 1):
        lo = i - seq_len + 1
        hi = i + k
        win_days = days[lo:hi]
        win_ts = ts[lo:hi]
        if np.any(win_days != win_days[0]):
            n_rejected_day += 1
            continue
        dts = np.diff(win_ts)
        if np.any(dts <= 0):
            n_rejected_order += 1
            continue
        if np.any(dts > max_gap_seconds + 1e-6):
            n_rejected_gap += 1
            continue
        xs.append(states[lo:i + 1])
        ys.append(next_states[i:i + k])
        out_days.append(win_days[0])
        out_ts.append(float(ts[i]))
        if stages is not None:
            out_stage.append(stages[i + 1: i + 1 + k])

    n_kept = len(xs)
    dim = int(states.shape[1]) if states.ndim == 2 else 0
    if n_kept == 0:
        x = np.zeros((0, seq_len, dim), dtype=np.float32)
        y = np.zeros((0, k, dim), dtype=np.float32)
        day_arr = np.array([], dtype=DAY_ID_DTYPE)
        ts_arr = np.zeros((0,), dtype=np.float64)
        stage_arr = np.zeros((0, k), dtype=np.int32)
    else:
        x = np.stack(xs).astype(np.float32)
        y = np.stack(ys).astype(np.float32)
        day_arr = np.array(out_days, dtype=DAY_ID_DTYPE)
        ts_arr = np.array(out_ts, dtype=np.float64)
        stage_arr = np.stack(out_stage).astype(np.int32) if out_stage else np.zeros((0, k), dtype=np.int32)

    out = {
        "x": x,
        "y_true": y,
        "day_id": day_arr,
        "timestamps": ts_arr,
        "n_kept": n_kept,
        "n_rejected_day": n_rejected_day,
        "n_rejected_gap": n_rejected_gap,
        "n_rejected_order": n_rejected_order,
        "seq_len": seq_len,
        "k": k,
    }
    if stages is not None:
        out["y_stage"] = stage_arr
    return out


def summarize_kstep(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Per-horizon and pooled next-state metrics. Arrays (n, k, dim)."""
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch {y_true.shape} vs {y_pred.shape}")
    n, k, dim = y_true.shape
    by_k = []
    for ki in range(k):
        block = next_state_metrics(y_true[:, ki, :], y_pred[:, ki, :])
        block["k"] = ki + 1
        by_k.append(block)
    overall = next_state_metrics(y_true.reshape(-1, dim), y_pred.reshape(-1, dim))
    return {"n": int(n), "k": int(k), "dim": int(dim), "by_k": by_k, "overall": overall}


def compare_kstep(y_true: np.ndarray, lstm_pred: np.ndarray, persist_pred: np.ndarray) -> dict:
    lstm = summarize_kstep(y_true, lstm_pred)
    persist = summarize_kstep(y_true, persist_pred)
    p_mse = persist["overall"]["mse"]
    l_mse = lstm["overall"]["mse"]
    reduction = float((p_mse - l_mse) / p_mse) if p_mse else 0.0
    by_k = []
    for a, b in zip(lstm["by_k"], persist["by_k"]):
        pm, lm = b["mse"], a["mse"]
        by_k.append({
            "k": a["k"],
            "lstm": a,
            "persist": b,
            "mse_reduction_vs_persist": float((pm - lm) / pm) if pm else 0.0,
        })
    return {
        "n": lstm["n"],
        "k": lstm["k"],
        "lstm_overall": lstm["overall"],
        "persist_overall": persist["overall"],
        "mse_reduction_vs_persist": reduction,
        "by_k": by_k,
    }
