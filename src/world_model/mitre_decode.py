"""Decode 32-d network states into SIH MITRE ATT&CK stages.

Fits a train-only multinomial logistic regression on scaled S_t.
Does not change or retrain the LSTM: rolled-out Ŝ_{t+k} is decoded after
the world model has already produced it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score

from src.world_model.labels import STAGE_NAMES, technique_for_stage

N_STAGES = len(STAGE_NAMES)


@dataclass
class StageDecoder:
    clf: LogisticRegression

    def predict(self, states: np.ndarray) -> np.ndarray:
        """states: (n, dim) or (n, k, dim) → int stage_id with the same leading shape."""
        arr, lead = _flatten_states(states)
        pred = self.clf.predict(arr).astype(np.int32)
        return pred.reshape(lead) if lead else pred

    def predict_proba(self, states: np.ndarray) -> np.ndarray:
        """Return (..., n_stages) probabilities aligned to STAGE_NAMES order."""
        arr, lead = _flatten_states(states)
        raw = self.clf.predict_proba(arr)
        full = np.zeros((len(arr), N_STAGES), dtype=np.float32)
        for col, sid in enumerate(self.clf.classes_):
            full[:, int(sid)] = raw[:, col]
        if lead:
            return full.reshape((*lead, N_STAGES))
        return full


def _flatten_states(states: np.ndarray) -> tuple[np.ndarray, tuple[int, ...] | None]:
    arr = np.asarray(states, dtype=np.float32)
    if arr.ndim == 2:
        return arr, None
    if arr.ndim == 3:
        n, k, dim = arr.shape
        return arr.reshape(n * k, dim), (n, k)
    raise ValueError(f"states must be (n, dim) or (n, k, dim), got {arr.shape}")


def fit_stage_decoder(train_states: np.ndarray, train_stage_ids: np.ndarray) -> StageDecoder:
    X = np.asarray(train_states, dtype=np.float32)
    y = np.asarray(train_stage_ids).astype(int)
    if X.ndim != 2:
        raise ValueError(f"train_states must be (n, dim), got {X.shape}")
    if len(X) != len(y):
        raise ValueError("train_states and train_stage_ids length mismatch")
    if len(np.unique(y)) < 2:
        raise ValueError("Need at least two stage classes in train to fit a decoder")
    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs",
    )
    clf.fit(X, y)
    return StageDecoder(clf)


def stage_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true).astype(int).ravel()
    y_pred = np.asarray(y_pred).astype(int).ravel()
    labels = list(range(N_STAGES))
    per = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    rec = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    per_stage = {}
    for i, name in enumerate(STAGE_NAMES):
        tid, tname = technique_for_stage(name)
        per_stage[name] = {
            "f1": float(per[i]),
            "recall": float(rec[i]),
            "support": int((y_true == i).sum()),
            "pred_count": int((y_pred == i).sum()),
            "technique_id": tid,
            "technique_name": tname,
        }
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "n": int(len(y_true)),
        "per_stage": per_stage,
        "confusion": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def compare_stage_rollout(
    y_stage: np.ndarray,
    lstm_states: np.ndarray,
    persist_states: np.ndarray,
    true_states: np.ndarray,
    decoder: StageDecoder,
) -> dict:
    """Score LSTM / persist / oracle decodes of K-step states against true stage_id."""
    lstm_hat = decoder.predict(lstm_states)
    persist_hat = decoder.predict(persist_states)
    oracle_hat = decoder.predict(true_states)
    lstm = stage_metrics(y_stage, lstm_hat)
    persist = stage_metrics(y_stage, persist_hat)
    oracle = stage_metrics(y_stage, oracle_hat)
    k = int(y_stage.shape[1]) if y_stage.ndim == 2 else 1
    by_k = []
    if y_stage.ndim == 2:
        for ki in range(k):
            by_k.append({
                "k": ki + 1,
                "lstm": stage_metrics(y_stage[:, ki], lstm_hat[:, ki]),
                "persist": stage_metrics(y_stage[:, ki], persist_hat[:, ki]),
                "oracle": stage_metrics(y_stage[:, ki], oracle_hat[:, ki]),
            })
    return {
        "n": int(y_stage.shape[0]) if y_stage.ndim >= 1 else 0,
        "k": k,
        "lstm": lstm,
        "persist": persist,
        "oracle": oracle,
        "by_k": by_k,
        "beats_persist_macro_f1": bool(lstm["macro_f1"] >= persist["macro_f1"]),
        "beats_persist_accuracy": bool(lstm["accuracy"] >= persist["accuracy"]),
    }
