"""Temporal loaders for (S_t, S_{t+1}, attack_within_k) arrays."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np

# SIH multi-day experiment (YYYY-MM-DD). Override via CLI.
DEFAULT_TRAIN_DAYS = (
    "2018-02-14",
    "2018-02-15",
    "2018-02-16",
    "2018-02-22",
    "2018-02-28",
)
DEFAULT_VAL_DAYS = ("2018-02-23",)
DEFAULT_TEST_DAYS = (
    "2018-02-21",
    "2018-03-01",
    "2018-03-02",
)

_ISO_DAY = re.compile(r"^\d{4}-\d{2}-\d{2}$")
# CIC filenames: Wednesday-14-02-2018_TrafficForML_...
_DMY = re.compile(r"(\d{2})-(\d{2})-(\d{4})")

_STACK_KEYS = (
    "states",
    "next_states",
    "attack_now",
    "attack_within_k",
    "infiltration_within_k",
    "pre_attack",
    "stage_id",
    "technique_id",
    "timestamps",
    "window_ids",
    "day_id",
    "source_file",
)


@dataclass
class Scaler:
    mean: np.ndarray
    std: np.ndarray

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def inverse(self, x: np.ndarray) -> np.ndarray:
        return x * self.std + self.mean

    def save(self, path: Path) -> None:
        np.savez(path, mean=self.mean, std=self.std)

    @classmethod
    def load(cls, path: Path) -> "Scaler":
        z = np.load(path)
        return cls(mean=z["mean"], std=z["std"])


@dataclass
class SequenceStats:
    n_kept: int = 0
    n_rejected_day: int = 0
    n_rejected_gap: int = 0
    n_rejected_order: int = 0


def canonical_day_id(token: str) -> str:
    """Map a CIC filename, stem, or ISO date to YYYY-MM-DD."""
    text = str(token).strip()
    if _ISO_DAY.fullmatch(text):
        return text
    m = _DMY.search(text.replace("_", "-"))
    if not m:
        raise ValueError(f"Cannot parse day identifier from {token!r}")
    dd, mm, yyyy = m.group(1), m.group(2), m.group(3)
    return f"{yyyy}-{mm}-{dd}"


def fit_scaler(train_states: np.ndarray) -> Scaler:
    mean = train_states.mean(axis=0).astype(np.float64)
    std = train_states.std(axis=0).astype(np.float64)
    std[std < 1e-8] = 1.0
    return Scaler(mean=mean, std=std)


def temporal_split(n: int, train=0.70, val=0.15) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Contiguous time split: train | val | test. No shuffle. Single-day fallback."""
    n_train = int(n * train)
    n_val = int(n * val)
    idx = np.arange(n)
    return idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:]


def day_split(
    day_ids: np.ndarray,
    train_days: Sequence[str],
    val_days: Sequence[str],
    test_days: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Index rows by explicit calendar days. Unlisted days are dropped."""
    tr_set = {canonical_day_id(d) for d in train_days}
    va_set = {canonical_day_id(d) for d in val_days}
    te_set = {canonical_day_id(d) for d in test_days}
    overlap = (tr_set & va_set) | (tr_set & te_set) | (va_set & te_set)
    if overlap:
        raise ValueError(f"train/val/test days overlap: {sorted(overlap)}")
    canon = np.array([canonical_day_id(d) for d in day_ids])
    train_i = np.flatnonzero(np.isin(canon, list(tr_set)))
    val_i = np.flatnonzero(np.isin(canon, list(va_set)))
    test_i = np.flatnonzero(np.isin(canon, list(te_set)))
    return train_i, val_i, test_i


def concat_window_bundles(bundles: Sequence[dict]) -> dict:
    """Stack per-day processed pairs. Never used on raw flow DataFrames."""
    if not bundles:
        raise ValueError("No window bundles to concatenate")
    out: dict = {}
    for key in _STACK_KEYS:
        if key not in bundles[0]:
            continue
        out[key] = np.concatenate([b[key] for b in bundles], axis=0)
    for key in ("feature_names", "window_seconds", "horizon_k"):
        out[key] = bundles[0][key]
    return out


def make_sequences(
    states: np.ndarray,
    next_states: np.ndarray,
    labels: np.ndarray,
    seq_len: int,
    timestamps: np.ndarray | None = None,
    day_ids: np.ndarray | None = None,
    window_seconds: float = 5.0,
    max_gap_seconds: float = 15.0,
    return_stats: bool = False,
):
    """
    Build length-`seq_len` histories.

    When timestamps/day_ids are provided, a sequence is kept only if:
      - every step shares the same day_id
      - timestamps are strictly increasing
      - each adjacent delta is <= max_gap_seconds (default 15s = 3 bins)
    """
    del window_seconds  # documented spacing; max_gap_seconds is the hard limit
    if len(states) < seq_len:
        raise ValueError(f"Need at least {seq_len} rows, got {len(states)}")

    if timestamps is None and day_ids is None:
        xs, ys, yc = [], [], []
        for i in range(seq_len - 1, len(states)):
            xs.append(states[i - seq_len + 1: i + 1])
            ys.append(next_states[i])
            yc.append(labels[i])
        x = np.stack(xs).astype(np.float32)
        y_next = np.stack(ys).astype(np.float32)
        y_atk = np.asarray(yc, dtype=np.float32)
        stats = SequenceStats(n_kept=len(x))
        seq_days = np.array(["unknown"] * len(x), dtype="U10")
        if return_stats:
            return x, y_next, y_atk, stats, seq_days
        return x, y_next, y_atk

    ts = np.asarray(timestamps, dtype=np.float64)
    if len(ts) != len(states):
        raise ValueError("timestamps length must match states")
    if day_ids is None:
        days = np.array(["unknown"] * len(states), dtype="U16")
    else:
        days = np.array([canonical_day_id(d) for d in day_ids])

    xs, ys, yc, seq_days = [], [], [], []
    stats = SequenceStats()
    for i in range(seq_len - 1, len(states)):
        sl = slice(i - seq_len + 1, i + 1)
        win_days = days[sl]
        win_ts = ts[sl]
        if np.any(win_days != win_days[0]):
            stats.n_rejected_day += 1
            continue
        dts = np.diff(win_ts)
        if np.any(dts <= 0):
            stats.n_rejected_order += 1
            continue
        if np.any(dts > max_gap_seconds + 1e-6):
            stats.n_rejected_gap += 1
            continue
        xs.append(states[sl])
        ys.append(next_states[i])
        yc.append(labels[i])
        seq_days.append(win_days[0])

    stats.n_kept = len(xs)
    if not xs:
        if not return_stats:
            raise ValueError(
                "No valid sequences after day-boundary / timestamp-gap filters "
                f"(rejected day={stats.n_rejected_day} gap={stats.n_rejected_gap} "
                f"order={stats.n_rejected_order})"
            )
        x = np.zeros((0, seq_len, states.shape[1]), dtype=np.float32)
        y_next = np.zeros((0, states.shape[1]), dtype=np.float32)
        y_atk = np.zeros((0,), dtype=np.float32)
        seq_day_arr = np.array([], dtype="U10")
        return x, y_next, y_atk, stats, seq_day_arr
    x = np.stack(xs).astype(np.float32)
    y_next = np.stack(ys).astype(np.float32)
    y_atk = np.asarray(yc, dtype=np.float32)
    seq_day_arr = np.array(seq_days, dtype="U10")
    if return_stats:
        return x, y_next, y_atk, stats, seq_day_arr
    return x, y_next, y_atk


def _day_from_unix(ts: float) -> str:
    return datetime.fromtimestamp(float(ts), tz=timezone.utc).strftime("%Y-%m-%d")


def load_npz(path: Path) -> dict:
    z = np.load(path, allow_pickle=True)
    order = np.argsort(z["timestamps"])
    ts = z["timestamps"][order]
    out = {
        "states": z["states"][order].astype(np.float32),
        "next_states": z["next_states"][order].astype(np.float32),
        "attack_within_k": z["attack_within_k"][order].astype(np.float32),
        "timestamps": ts,
        "feature_names": z["feature_names"],
    }
    names = set(z.files)
    if "day_id" in names:
        out["day_id"] = np.array([canonical_day_id(d) for d in z["day_id"][order]])
    else:
        out["day_id"] = np.array([_day_from_unix(t) for t in ts], dtype="U10")
    if "source_file" in names:
        out["source_file"] = np.asarray(z["source_file"][order], dtype=object)
    else:
        out["source_file"] = np.array(["unknown"] * len(ts), dtype=object)
    for key in ("attack_now", "window_ids", "infiltration_within_k", "pre_attack", "stage_id", "technique_id"):
        if key in names:
            out[key] = z[key][order]
    return out
