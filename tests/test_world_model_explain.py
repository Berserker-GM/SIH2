#!/usr/bin/env python3
"""Why-explanations for the frozen world model: no retrain, no architecture change."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.world_model.cic_schema import STATE_FEATURE_ORDER
from src.world_model.labels import STAGE_ID, STAGE_NAMES, technique_for_stage
from src.world_model.mitre_decode import fit_stage_decoder
from src.world_model.model import LSTMWorldModel

results = []


def check(name: str, condition: bool, detail: str = "") -> None:
    results.append((name, condition))
    icon = "PASS" if condition else "FAIL"
    extra = f"  ({detail})" if detail else ""
    print(f"  {icon}  {name}{extra}")


print("\n-- Attack saliency shape ----------------------------------------------")
from src.world_model.explain import attack_saliency, explain_forecast, stage_contributions, top_features

model = LSTMWorldModel(input_dim=32, hidden_dim=64, num_layers=2, dropout=0.2)
model.eval()
rng = np.random.default_rng(1)
x = rng.normal(size=(3, 8, 32)).astype(np.float32)
sal = attack_saliency(model, x)
check("saliency (n, 32)", sal.shape == (3, 32))
check("saliency non-negative", np.all(sal >= 0))
check("saliency not all zero", float(sal.sum()) > 0)

src = (PROJECT_ROOT / "src" / "world_model" / "model.py").read_text(encoding="utf-8")
check("model.py still has no explain/shap head", "shap" not in src.casefold() and "explain" not in src.casefold())


print("\n-- Stage linear contributions -----------------------------------------")
# Feature 2 separates lateral_movement from benign.
X = np.zeros((80, 32), dtype=np.float32)
y = np.zeros(80, dtype=np.int32)
X[40:, 2] = 5.0
y[40:] = STAGE_ID["lateral_movement"]
dec = fit_stage_decoder(X, y)
state = np.zeros(32, dtype=np.float32)
state[2] = 5.0
contrib = stage_contributions(dec, state, STAGE_ID["lateral_movement"])
check("contrib is 32-d", contrib.shape == (32,))
check("feature 2 dominates stage contrib", int(np.argmax(np.abs(contrib))) == 2)

ranked = top_features(contrib, STATE_FEATURE_ORDER, k=3)
check("top feature is dest_port_entropy", ranked[0]["name"] == "dest_port_entropy")
check("top_features has signed score", "score" in ranked[0] and "direction" in ranked[0])


print("\n-- explain_forecast payload -------------------------------------------")
hist = np.zeros((8, 32), dtype=np.float32)
hist[:, 2] = np.linspace(0, 4, 8)
bundle = explain_forecast(model, dec, hist, k=6, attack_threshold=0.15)
check("has attack_probability", 0.0 <= bundle["attack_probability"] <= 1.0)
check("has something_bad flag", isinstance(bundle["something_bad"], bool))
check("threshold is 0.15", bundle["attack_threshold"] == 0.15)
check("stage is a SIH name", bundle["stage"] in STAGE_NAMES)
tid, tname = technique_for_stage(bundle["stage"])
check("technique matches stage table", bundle["technique_id"] == tid)
check("forecast states (6, 32)", np.asarray(bundle["forecast_states"]).shape == (6, 32))
check("why_attack is list", isinstance(bundle["why_attack"], list) and len(bundle["why_attack"]) >= 1)
check("why_stage is list", isinstance(bundle["why_stage"], list) and len(bundle["why_stage"]) >= 1)
check("why_change is list", isinstance(bundle["why_change"], list) and len(bundle["why_change"]) >= 1)
check("narrative answers why", "why" in bundle["narrative"].casefold() or "because" in bundle["narrative"].casefold())
check("2x64 LSTM unchanged", model.lstm.num_layers == 2 and model.lstm.hidden_size == 64)


print("\n-- CopyLast logit=0 → zero saliency -----------------------------------")


class ZeroAttack(nn.Module):
    def forward(self, x: torch.Tensor):
        last = x[:, -1, :]
        return last, torch.zeros(x.size(0), device=x.device, dtype=x.dtype)


zsal = attack_saliency(ZeroAttack(), x)
check("zero attack logit → zero saliency", float(zsal.sum()) == 0.0)


print("\n-- Summary ------------------------------------------------------------")
passed = sum(1 for _, ok in results if ok)
failed = sum(1 for _, ok in results if not ok)
print(f"\n  Total: {len(results)}   Passed: {passed}   Failed: {failed}\n")
if failed:
    sys.exit(1)
print("  World-model explanation helpers OK.\n")
