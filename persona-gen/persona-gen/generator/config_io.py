"""
Persona config loading/validation.

Every config is filled against DEFAULT_CONFIG so downstream code never needs
per-persona branching (e.g. benign personas still have a `phases` list and a
neutral `packet_fields`, just with values that make them behave as steady
background traffic).
"""
import json
from pathlib import Path

DEFAULT_CONFIG = {
    "persona_id": None,
    "family": "benign",
    "mitre_technique": None,
    "adaptive": False,
    "base_seed": 0,
    "timing": {"interarrival_dist": "exponential", "lambda": 2.0, "jitter_pct": 0.15},
    "flags": {
        "syn_ratio": 0.15, "ack_ratio": 0.55, "fin_ratio": 0.10,
        "rst_ratio": 0.05, "psh_ratio": 0.15, "urg_ratio": 0.01,
        "tcp_ratio": 0.9,
        "down_up_ratio_mean": 1.0, "down_up_ratio_std": 0.2,
        "bidirectional_ratio_mean": 0.5,
        "bwd_pkt_len_ratio": 1.0,
        "init_fwd_win_mean": 8192, "init_bwd_win_mean": 8192, "init_win_std": 1024,
        "port_scan_score_base": 0.0, "port_scan_score_gain": 0.2,
    },
    "ports": {"pool": list(range(1024, 65535)), "selection": "randomized"},
    "payload": {"size_dist": "lognormal", "mean": 512, "std": 128},
    "packet_fields": {"ttl_base": 64, "ttl_jitter": 2, "fragment_rate": 0.0, "retransmit_rate": 0.01},
    "phases": [
        {"name": "steady", "mitre_stage": "benign", "technique_id": None,
         "dwell_windows": [100000, 100000], "next": None}
    ],
    "noise": {"stage_overlap_pct": 0.0, "blend_alpha": 0.3},
}

REQUIRED_TOP_LEVEL = [
    "persona_id", "family", "adaptive", "base_seed", "timing",
    "flags", "ports", "payload", "packet_fields", "phases", "noise",
]

VALID_FAMILIES = {
    "benign", "initial_access", "lateral_movement", "impact",
    "reconnaissance", "command_and_control", "exfiltration",
}


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _validate(cfg: dict):
    for k in REQUIRED_TOP_LEVEL:
        assert k in cfg, f"config for {cfg.get('persona_id')} missing required key: {k}"
    assert cfg["family"] in VALID_FAMILIES, f"invalid family {cfg['family']} for {cfg['persona_id']}"
    assert cfg["phases"], f"{cfg['persona_id']} has no phases"
    names = set()
    for phase in cfg["phases"]:
        for req in ("name", "mitre_stage", "dwell_windows", "next"):
            assert req in phase, f"{cfg['persona_id']} phase missing '{req}': {phase}"
        assert phase["mitre_stage"] in MITRE_STAGES_SET, (
            f"{cfg['persona_id']} phase '{phase['name']}' has invalid mitre_stage {phase['mitre_stage']}"
        )
        names.add(phase["name"])
    for phase in cfg["phases"]:
        nxt = phase["next"]
        assert nxt is None or nxt in names, (
            f"{cfg['persona_id']} phase '{phase['name']}' points to unknown next phase '{nxt}'"
        )


from .schema import MITRE_STAGES  # noqa: E402
MITRE_STAGES_SET = set(MITRE_STAGES)


def load_config(path) -> dict:
    with open(path) as f:
        raw = json.load(f)
    merged = _deep_merge(DEFAULT_CONFIG, raw)
    _validate(merged)
    return merged


def load_all_configs(config_dir) -> dict:
    config_dir = Path(config_dir)
    out = {}
    for p in sorted(config_dir.glob("*.json")):
        cfg = load_config(p)
        out[cfg["persona_id"]] = cfg
    return out
