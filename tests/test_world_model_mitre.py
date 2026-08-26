#!/usr/bin/env python3
"""MITRE stage decode from forecast states: no LSTM retrain, no architecture change."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.world_model.dataset import load_npz
from src.world_model.labels import STAGE_ID, STAGE_NAMES, map_label
from src.world_model.model import LSTMWorldModel
from src.world_model.prepare_dataset import generate_demo_flows
from src.world_model.rollout import collect_kstep_windows, persist_rollout, rollout_closed_loop
from src.world_model.windows import WindowConfig, build_state_windows

results = []


def check(name: str, condition: bool, detail: str = "") -> None:
    results.append((name, condition))
    icon = "PASS" if condition else "FAIL"
    extra = f"  ({detail})" if detail else ""
    print(f"  {icon}  {name}{extra}")


class CopyLast(nn.Module):
    def forward(self, x: torch.Tensor):
        last = x[:, -1, :]
        return last, torch.zeros(x.size(0), device=x.device)


print("\n-- SIH stage table ----------------------------------------------------")
check("seven SIH stages", len(STAGE_NAMES) == 7)
check("benign is 0", STAGE_ID["benign"] == 0)
check("exfiltration present", "exfiltration" in STAGE_NAMES)
from src.world_model.labels import technique_for_stage
tid, tname = technique_for_stage("lateral_movement")
check("lateral_movement → T1021", tid == "T1021")
check("C2 → T1071", technique_for_stage("command_and_control")[0] == "T1071")
check("impact → T1498", technique_for_stage("impact")[0] == "T1498")
check("benign has empty technique", technique_for_stage("benign")[0] == "")


print("\n-- load_npz keeps stage_id --------------------------------------------")
df = generate_demo_flows(n_windows=24, flows_per_window=3)
bundle = build_state_windows(df, WindowConfig(window_seconds=5, horizon_k=6))
with tempfile.TemporaryDirectory() as td:
    path = Path(td) / "with_stage.npz"
    np.savez(path, **bundle)
    loaded = load_npz(path)
    check("stage_id loaded", "stage_id" in loaded)
    check("stage_id length matches states", len(loaded["stage_id"]) == len(loaded["states"]))
    check("stage ids are known", set(loaded["stage_id"]).issubset(set(range(len(STAGE_NAMES)))))
    check("demo has more than benign", len(set(loaded["stage_id"].tolist())) >= 2)


print("\n-- load_npz without stage_id still works ------------------------------")
with tempfile.TemporaryDirectory() as td:
    path = Path(td) / "old.npz"
    np.savez(
        path,
        states=bundle["states"],
        next_states=bundle["next_states"],
        attack_within_k=bundle["attack_within_k"],
        timestamps=bundle["timestamps"],
        feature_names=bundle["feature_names"],
    )
    old = load_npz(path)
    check("old npz has no stage_id", "stage_id" not in old)


print("\n-- K-step future stage_id is S_{t+1} .. S_{t+k} -----------------------")
batch = collect_kstep_windows(
    bundle["states"],
    bundle["next_states"],
    bundle["timestamps"],
    bundle["day_id"],
    seq_len=8,
    k=6,
    max_gap_seconds=15.0,
    stage_ids=bundle["stage_id"],
)
check("y_stage present", "y_stage" in batch)
check("y_stage shape (n, k)", batch["y_stage"].shape == (batch["n_kept"], 6))
check("kept some staged windows", batch["n_kept"] >= 1)

# Reconstruct each kept row from timestamp; true stage of S_{t+j} is stage_id[i+j]
# (states[i+j] == next_states[i+j-1] on this consecutive demo).
ts_to_i = {float(t): i for i, t in enumerate(bundle["timestamps"])}
aligned = True
for row, t_i in enumerate(batch["timestamps"]):
    i = ts_to_i[float(t_i)]
    expected = bundle["stage_id"][i + 1: i + 1 + 6]
    if not np.array_equal(batch["y_stage"][row], expected):
        aligned = False
        break
check("y_stage[i, j] == stage_id[i+1+j]", aligned)

# Without stage_ids, k-step collect must not require the extra last pair.
plain = collect_kstep_windows(
    bundle["states"],
    bundle["next_states"],
    bundle["timestamps"],
    bundle["day_id"],
    seq_len=8,
    k=6,
)
check("no y_stage unless asked", "y_stage" not in plain)
check("staged collect never keeps more than plain", batch["n_kept"] <= plain["n_kept"])


print("\n-- Stage decoder fits train-only 32-d states --------------------------")
from src.world_model.mitre_decode import StageDecoder, fit_stage_decoder, stage_metrics

rng = np.random.default_rng(0)
# Three linearly separable blobs in 32-d.
means = np.array([
    np.zeros(32),
    np.ones(32) * 4,
    np.concatenate([np.ones(16) * -4, np.ones(16) * 4]),
], dtype=np.float32)
y_ids = np.array([STAGE_ID["benign"], STAGE_ID["initial_access"], STAGE_ID["lateral_movement"]])
X_train, y_train = [], []
for sid, mu in zip(y_ids, means):
    X_train.append(rng.normal(mu, 0.2, size=(40, 32)).astype(np.float32))
    y_train.append(np.full(40, sid, dtype=np.int32))
X_train = np.concatenate(X_train)
y_train = np.concatenate(y_train)
decoder = fit_stage_decoder(X_train, y_train)
check("decoder is StageDecoder", isinstance(decoder, StageDecoder))
pred = decoder.predict(means)
check("recovers three train stages", np.array_equal(pred, y_ids))

X_traj = np.stack([means, means], axis=0)  # (n=2, k=3? wait we need k dim)
# (n, k, 32): two trajectories of length 3
traj = np.stack([np.stack([means[0], means[1], means[2]]), np.stack([means[2], means[1], means[0]])])
traj_pred = decoder.predict(traj)
check("predict accepts (n, k, 32)", traj_pred.shape == (2, 3))
check("traj row0 stages", np.array_equal(traj_pred[0], y_ids))

m = stage_metrics(y_ids, pred)
check("perfect accuracy on prototypes", m["accuracy"] == 1.0)
check("macro_f1 present", "macro_f1" in m)
check("per_stage has 7 names", set(m["per_stage"]) == set(STAGE_NAMES))


print("\n-- Closed-loop CopyLast decode equals persist decode ------------------")
dec_states = decoder.predict(persist_rollout(means[:2], k=4))
copy_states, _ = rollout_closed_loop(
    CopyLast(),
    torch.from_numpy(np.repeat(means[:2, None, :], 8, axis=1)),
    k=4,
)
check("CopyLast decode matches persist", np.array_equal(decoder.predict(copy_states), dec_states))


print("\n-- LSTM architecture still 2x64 / 32-d --------------------------------")
model = LSTMWorldModel(input_dim=32, hidden_dim=64, num_layers=2, dropout=0.2)
check("2-layer LSTM", model.lstm.num_layers == 2 and model.lstm.hidden_size == 64)
check("next head 32-d", model.next_head[-1].out_features == 32)
src = (PROJECT_ROOT / "src" / "world_model" / "model.py").read_text(encoding="utf-8")
check("model.py still two heads only", "stage_head" not in src and "mitre" not in src.casefold())


print("\n-- Unknown CIC labels still map --------------------------------------")
check("bot → command_and_control", map_label("Bot")["stage"] == "command_and_control")
check("HOIC → impact", map_label("DDoS attack-HOIC")["stage"] == "impact")
check("exfil stage reserved", STAGE_ID["exfiltration"] == 5)


print("\n-- Summary ------------------------------------------------------------")
passed = sum(1 for _, ok in results if ok)
failed = sum(1 for _, ok in results if not ok)
print(f"\n  Total: {len(results)}   Passed: {passed}   Failed: {failed}\n")
if failed:
    sys.exit(1)
print("  MITRE stage decode helpers OK.\n")
