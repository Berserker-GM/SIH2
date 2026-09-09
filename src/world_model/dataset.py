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

# syn:{persona_id}:{run_id}  — one synthetic run == one calendar "day"
# U128: long persona+run ids (e.g. recon_service_discovery_v1__run_000) exceed U64.
SYNTH_DAY_PREFIX = "syn:"
DAY_ID_DTYPE = "U128"
SOURCE_DTYPE = "U16"

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
    "source",
    "persona_id",
    "run_id",
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
    """Map a CIC filename, stem, ISO date, or synthetic run id to a split key.

    Synthetic runs use `syn:{persona_id}:{run_id}` and are treated exactly like
    one calendar day: sequences never cross a run boundary.
    """
    text = str(token).strip()
    if not text:
        raise ValueError("empty day identifier")
    if _ISO_DAY.fullmatch(text):
        return text
    if text.startswith(SYNTH_DAY_PREFIX):
        return text
    m = _DMY.search(text.replace("_", "-"))
    if m:
        dd, mm, yyyy = m.group(1), m.group(2), m.group(3)
        return f"{yyyy}-{mm}-{dd}"
    return text


def make_synth_day_id(persona_id: str, run_id: str) -> str:
    p = str(persona_id).strip()
    r = str(run_id).strip()
    if not p or not r:
        raise ValueError("persona_id and run_id are required")
    if ":" in p or "/" in p or "\\" in p:
        raise ValueError(f"persona_id must not contain ':' or slashes: {p!r}")
    if "/" in r or "\\" in r:
        raise ValueError(f"run_id must not contain slashes: {r!r}")
    return f"{SYNTH_DAY_PREFIX}{p}:{r}"


def parse_synth_day_id(day_id: str) -> tuple[str, str] | None:
    text = canonical_day_id(day_id)
    if not text.startswith(SYNTH_DAY_PREFIX):
        return None
    rest = text[len(SYNTH_DAY_PREFIX):]
    persona, sep, run = rest.partition(":")
    if not sep or not persona or not run:
        raise ValueError(f"malformed synthetic day_id: {day_id!r}")
    return persona, run


def is_synthetic_day(day_id: str) -> bool:
    return str(day_id).startswith(SYNTH_DAY_PREFIX)


def split_synthetic_days(day_ids: Sequence[str]) -> tuple[list[str], list[str], list[str]]:
    """Assign each syn:{persona}:{run} to train/val/test. Never splits a run.

    Per persona, sorted by run_id:
      1 run  → train
      2 runs → train, test
      3+     → all but last two train, second-last val, last test
    """
    from collections import defaultdict

    by_persona: dict[str, list[str]] = defaultdict(list)
    unique = sorted({canonical_day_id(d) for d in day_ids if is_synthetic_day(d)})
    for d in unique:
        parsed = parse_synth_day_id(d)
        if parsed is None:
            continue
        by_persona[parsed[0]].append(d)
    train: list[str] = []
    val: list[str] = []
    test: list[str] = []
    for persona in sorted(by_persona):
        runs = sorted(by_persona[persona])
        n = len(runs)
        if n == 1:
            train.extend(runs)
        elif n == 2:
            train.append(runs[0])
            test.append(runs[1])
        else:
            train.extend(runs[:-2])
            val.append(runs[-2])
            test.append(runs[-1])
    return train, val, test


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
        seq_days = np.array(["unknown"] * len(x), dtype=DAY_ID_DTYPE)
        if return_stats:
            return x, y_next, y_atk, stats, seq_days
        return x, y_next, y_atk

    ts = np.asarray(timestamps, dtype=np.float64)
    if len(ts) != len(states):
        raise ValueError("timestamps length must match states")
    if day_ids is None:
        days = np.array(["unknown"] * len(states), dtype=DAY_ID_DTYPE)
    else:
        days = np.array([canonical_day_id(d) for d in day_ids], dtype=DAY_ID_DTYPE)

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
        seq_day_arr = np.array([], dtype=DAY_ID_DTYPE)
        return x, y_next, y_atk, stats, seq_day_arr
    x = np.stack(xs).astype(np.float32)
    y_next = np.stack(ys).astype(np.float32)
    y_atk = np.asarray(yc, dtype=np.float32)
    seq_day_arr = np.array(seq_days, dtype=DAY_ID_DTYPE)
    if return_stats:
        return x, y_next, y_atk, stats, seq_day_arr
    return x, y_next, y_atk


def _day_from_unix(ts: float) -> str:
    return datetime.fromtimestamp(float(ts), tz=timezone.utc).strftime("%Y-%m-%d")


def load_npz(path: Path) -> dict:
    z = np.load(path, allow_pickle=True)
    names = set(z.files)
    ts_raw = np.asarray(z["timestamps"], dtype=np.float64)
    if "day_id" in names:
        day_raw = np.array([canonical_day_id(d) for d in z["day_id"]], dtype=DAY_ID_DTYPE)
    else:
        day_raw = np.array([_day_from_unix(t) for t in ts_raw], dtype=DAY_ID_DTYPE)
    # Sort by day then time so synthetic runs with overlapping clocks stay contiguous.
    order = np.lexsort((ts_raw, day_raw))
    ts = ts_raw[order]
    out = {
        "states": z["states"][order].astype(np.float32),
        "next_states": z["next_states"][order].astype(np.float32),
        "attack_within_k": z["attack_within_k"][order].astype(np.float32),
        "timestamps": ts,
        "feature_names": z["feature_names"],
        "day_id": day_raw[order],
    }
    n = len(ts)
    if "source_file" in names:
        out["source_file"] = np.asarray(z["source_file"][order], dtype=object)
    else:
        out["source_file"] = np.array(["unknown"] * n, dtype=object)
    if "source" in names:
        out["source"] = np.asarray(z["source"][order], dtype=SOURCE_DTYPE)
    else:
        out["source"] = np.array(
            ["synthetic" if is_synthetic_day(d) else "real" for d in out["day_id"]],
            dtype=SOURCE_DTYPE,
        )
    if "persona_id" in names:
        out["persona_id"] = np.asarray(z["persona_id"][order], dtype=DAY_ID_DTYPE)
    else:
        out["persona_id"] = np.array([""] * n, dtype=DAY_ID_DTYPE)
    if "run_id" in names:
        out["run_id"] = np.asarray(z["run_id"][order], dtype=DAY_ID_DTYPE)
    else:
        out["run_id"] = np.array([""] * n, dtype=DAY_ID_DTYPE)
    for key in ("attack_now", "window_ids", "infiltration_within_k", "pre_attack", "stage_id", "technique_id"):
        if key in names:
            out[key] = z[key][order]
    return out
