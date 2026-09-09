"""
PLACEHOLDER — mock of the pipeline dev's score_history().

This is NOT the real LSTM scorer. It exists only so the hill-climbing loop
(adaptive.py) can be developed and unit-tested in isolation right now. Per
the interface contract, the real function is an in-process Python call the
pipeline dev ships — swap the import in adaptive.py for it once confirmed;
do not ship this stub as the scoring path.

The heuristic below is a fixed, deterministic weighted combination of a few
"suspicious" feature columns, passed through a logistic, so the hill-climbing
loop has a real gradient-like signal to descend during local testing.
"""
import numpy as np

from .schema import FEATURE_COLUMNS

_IDX = {name: i for i, name in enumerate(FEATURE_COLUMNS)}

# Feature name -> weight. Purely illustrative; not calibrated against any
# real model. Positive weight = pushes attack_probability up.
_WEIGHTS = {
    "port_scan_score": 3.0,
    "syn_flag_sum": 0.015,
    "rst_flag_sum": 0.02,
    "dest_port_entropy": 0.3,
    "retransmit_count": 0.05,
    "ttl_variance": 0.15,
    "ip_fragment_flags": 0.1,
    "down_up_ratio_mean": -0.15,  # very low down/up (uplink-heavy, exfil-like) also matters; handled below
    "unique_dst_ports": 0.02,
}
_BIAS = -2.5


def score_history(x: np.ndarray, model_version: str = "v2") -> dict:
    """Mock of the real contract. x: shape (8, 32), most recent window last."""
    assert x.shape == (8, len(FEATURE_COLUMNS)), f"expected (8, {len(FEATURE_COLUMNS)}), got {x.shape}"

    recent = x[-1]  # most recent window dominates the score, like a real recurrent scorer would weight it
    trend = x.mean(axis=0)

    z = _BIAS
    for feat, w in _WEIGHTS.items():
        z += w * recent[_IDX[feat]]
    # exfiltration-style uplink skew: very low down_up_ratio_mean is itself suspicious
    dur = recent[_IDX["down_up_ratio_mean"]]
    if dur < 0.3:
        z += 1.2 * (0.3 - dur)
    # trend term: sustained elevated port_scan_score across the window is worse than a one-off spike
    z += 1.5 * trend[_IDX["port_scan_score"]]
    # beaconing-regularity signal: a near-perfectly periodic IAT and near-uniform payload
    # size across the window is itself a C2 tell, independent of volume — this is exactly
    # what jitter/payload-variance evasion is trying to break up.
    iat_mean = max(recent[_IDX["iat_mean"]], 1e-3)
    iat_regularity = max(0.0, 1.0 - (recent[_IDX["iat_std"]] / iat_mean))
    pkt_mean = max(recent[_IDX["pkt_len_mean"]], 1e-3)
    size_regularity = max(0.0, 1.0 - (recent[_IDX["pkt_len_std"]] / pkt_mean))
    z += 2.0 * iat_regularity + 1.0 * size_regularity

    attack_probability = float(1.0 / (1.0 + np.exp(-z)))
    something_bad = attack_probability > 0.5

    if recent[_IDX["port_scan_score"]] > 0.2:
        stage, technique = "reconnaissance", "T1595.001"
    elif dur < 0.3:
        stage, technique = "exfiltration", "T1041"
    elif recent[_IDX["syn_flag_sum"]] > recent[_IDX["ack_flag_sum"]]:
        stage, technique = "command_and_control", "T1071.001"
    else:
        stage, technique = "benign", None

    top = sorted(_WEIGHTS.items(), key=lambda kv: abs(kv[1] * recent[_IDX[kv[0]]]), reverse=True)[:3]
    why = [(name, float(w * recent[_IDX[name]])) for name, w in top]

    return {
        "attack_probability": attack_probability,
        "something_bad": something_bad,
        "stage": stage,
        "technique_id": technique,
        "why_attack_top_features": why,
        # Extra fields present on the real score_history() per the pipeline dev's v2 report.
        # Not meaningfully computed here — placeholders only, kept so calling code written
        # against the real signature doesn't break when running against this stub.
        "ood_score": None,
        "ood_flag": False,
        "ood_threshold": 6.0,
        "why_stage": [],
        "why_change": [],
        "narrative": None,
    }
