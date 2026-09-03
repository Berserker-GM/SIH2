"""
Per-window (5s) feature sampler.

Deterministic given an np.random.Generator seeded upstream — no LLM, no
network calls. This is the "controlled statistical realism" layer: every
one of the 32 contract features is derived from the persona/phase's
timing / flags / ports / payload / packet_fields knobs plus bounded noise,
rather than being hand-authored per persona.
"""
import numpy as np

from .schema import FEATURE_COLUMNS


def _resolve(persona_cfg: dict, phase: dict, key: str) -> dict:
    """Phase-level override wins; otherwise fall back to persona defaults."""
    return phase.get(key, persona_cfg[key])


def sample_window(persona_cfg: dict, phase: dict, rng: np.random.Generator) -> dict:
    timing = _resolve(persona_cfg, phase, "timing")
    flags = _resolve(persona_cfg, phase, "flags")
    ports = _resolve(persona_cfg, phase, "ports")
    payload = _resolve(persona_cfg, phase, "payload")
    pf = _resolve(persona_cfg, phase, "packet_fields")

    # --- flow volume for this 5s window ---
    jitter = timing.get("jitter_pct", 0.15)
    lam = timing["lambda"] * (1.0 + rng.uniform(-jitter, jitter))
    lam = max(lam, 1e-3)
    expected_flows = 5.0 * lam
    flow_count = int(rng.poisson(expected_flows))

    # --- destination ports ---
    pool = ports["pool"]
    pool_size = max(len(pool), 1)
    draws = None
    if flow_count == 0:
        unique_dst_ports = 0
        dest_port_entropy = 0.0
    elif ports["selection"] == "sequential":
        unique_dst_ports = int(min(flow_count, pool_size))
        dest_port_entropy = float(np.log2(unique_dst_ports)) if unique_dst_ports > 1 else 0.0
    else:
        draws = rng.integers(0, pool_size, size=flow_count)
        counts = np.bincount(draws)
        counts = counts[counts > 0]
        unique_dst_ports = int(len(counts))
        probs = counts / counts.sum()
        dest_port_entropy = float(-(probs * np.log2(probs)).sum())

    tcp_ratio = float(np.clip(flags.get("tcp_ratio", 0.9) + rng.normal(0, 0.03), 0.0, 1.0))
    udp_ratio = float(np.clip(1.0 - tcp_ratio + rng.normal(0, 0.02), 0.0, 1.0))

    # --- payload / packet sizes ---
    mean_size = max(payload["mean"], 1.0)
    std_size = max(payload["std"], 1.0)
    n_pkts_for_sizing = max(flow_count, 1)
    if payload["size_dist"] == "lognormal":
        sigma = float(np.sqrt(np.log(1 + (std_size / mean_size) ** 2)))
        mu = float(np.log(mean_size) - sigma ** 2 / 2)
        pkt_sizes = rng.lognormal(mu, sigma, size=n_pkts_for_sizing)
    else:
        pkt_sizes = rng.normal(mean_size, std_size, size=n_pkts_for_sizing)
    pkt_sizes = np.clip(pkt_sizes, 1.0, None)
    pkt_len_mean = float(pkt_sizes.mean())
    pkt_len_std = float(pkt_sizes.std())
    pkt_len_max = float(pkt_sizes.max())

    down_up_ratio_mean = float(max(0.0, flags.get("down_up_ratio_mean", 1.0)
                                    + rng.normal(0, flags.get("down_up_ratio_std", 0.2))))
    bidirectional_ratio = float(np.clip(flags.get("bidirectional_ratio_mean", 0.5) + rng.normal(0, 0.05), 0.0, 1.0))

    pkts_per_flow = max(1, int(rng.poisson(6)))
    pkts_fwd_sum = int(flow_count * pkts_per_flow)
    if down_up_ratio_mean > 0.02:
        pkts_bwd_sum = int(pkts_fwd_sum / down_up_ratio_mean)
    else:
        pkts_bwd_sum = int(pkts_fwd_sum * 0.01)
    bytes_fwd_sum = float(pkts_fwd_sum * pkt_len_mean)
    bytes_bwd_sum = float(pkts_bwd_sum * pkt_len_mean * flags.get("bwd_pkt_len_ratio", 1.0))

    # --- timing ---
    mean_interarrival = 1.0 / lam
    duration_mean = float(max(1e-3, mean_interarrival * rng.uniform(0.5, 1.5)))
    duration_max = float(duration_mean * rng.uniform(1.2, 3.0))
    iat_mean = float(max(1e-3, mean_interarrival * (1 + rng.normal(0, 0.1))))
    iat_std = float(abs(iat_mean * jitter * rng.uniform(0.5, 1.5)))
    iat_max = float(iat_mean + 3 * iat_std)

    # --- TCP flags ---
    def flag_sum(ratio_key, default):
        ratio = float(np.clip(flags.get(ratio_key, default) + rng.normal(0, 0.02), 0.0, 1.0))
        return int(flow_count * ratio), ratio

    syn_flag_sum, _ = flag_sum("syn_ratio", 0.15)
    ack_flag_sum, _ = flag_sum("ack_ratio", 0.55)
    fin_flag_sum, _ = flag_sum("fin_ratio", 0.10)
    rst_flag_sum, _ = flag_sum("rst_ratio", 0.05)
    psh_flag_sum, psh_ratio = flag_sum("psh_ratio", 0.15)
    urg_flag_sum, _ = flag_sum("urg_ratio", 0.01)
    fwd_psh_flags_sum = int(psh_flag_sum * rng.uniform(0.5, 1.0))

    init_fwd_win_mean = float(max(0.0, rng.normal(flags.get("init_fwd_win_mean", 8192),
                                                    flags.get("init_win_std", 1024))))
    init_bwd_win_mean = float(max(0.0, rng.normal(flags.get("init_bwd_win_mean", 8192),
                                                    flags.get("init_win_std", 1024))))

    port_scan_score = float(np.clip(
        flags.get("port_scan_score_base", 0.0)
        + (unique_dst_ports / pool_size) * flags.get("port_scan_score_gain", 0.2)
        + rng.normal(0, 0.03),
        0.0, 1.0,
    ))

    ttl_variance = float(max(0.0, rng.normal(pf.get("ttl_jitter", 2) ** 2, 0.5)))
    ip_fragment_flags = int(max(0, rng.poisson(flow_count * pf.get("fragment_rate", 0.0))))
    retransmit_count = int(max(0, rng.poisson(flow_count * pf.get("retransmit_rate", 0.01))))

    row = {
        "flow_count": flow_count,
        "unique_dst_ports": unique_dst_ports,
        "dest_port_entropy": round(dest_port_entropy, 4),
        "tcp_ratio": round(tcp_ratio, 4),
        "udp_ratio": round(udp_ratio, 4),
        "bytes_fwd_sum": round(bytes_fwd_sum, 2),
        "bytes_bwd_sum": round(bytes_bwd_sum, 2),
        "bidirectional_ratio": round(bidirectional_ratio, 4),
        "pkts_fwd_sum": pkts_fwd_sum,
        "pkts_bwd_sum": pkts_bwd_sum,
        "duration_mean": round(duration_mean, 4),
        "duration_max": round(duration_max, 4),
        "iat_mean": round(iat_mean, 4),
        "iat_std": round(iat_std, 4),
        "iat_max": round(iat_max, 4),
        "syn_flag_sum": syn_flag_sum,
        "ack_flag_sum": ack_flag_sum,
        "fin_flag_sum": fin_flag_sum,
        "rst_flag_sum": rst_flag_sum,
        "psh_flag_sum": psh_flag_sum,
        "urg_flag_sum": urg_flag_sum,
        "down_up_ratio_mean": round(down_up_ratio_mean, 4),
        "pkt_len_mean": round(pkt_len_mean, 2),
        "pkt_len_std": round(pkt_len_std, 2),
        "pkt_len_max": round(pkt_len_max, 2),
        "init_fwd_win_mean": round(init_fwd_win_mean, 2),
        "init_bwd_win_mean": round(init_bwd_win_mean, 2),
        "fwd_psh_flags_sum": fwd_psh_flags_sum,
        "port_scan_score": round(port_scan_score, 4),
        "ttl_variance": round(ttl_variance, 4),
        "ip_fragment_flags": ip_fragment_flags,
        "retransmit_count": retransmit_count,
    }
    assert set(row.keys()) == set(FEATURE_COLUMNS), "sampler output must match the 32-column contract exactly"
    return row


def blend_toward_benign(row: dict, benign_row: dict, alpha: float) -> dict:
    """Softens class separability: blend a fraction `alpha` of this window's
    features toward a benign reference profile, without touching the label.
    Used to avoid artificially clean (trivially separable) attack classes."""
    out = {}
    for k in FEATURE_COLUMNS:
        v, b = row[k], benign_row[k]
        blended = (1 - alpha) * v + alpha * b
        out[k] = int(round(blended)) if isinstance(v, int) else round(float(blended), 4)
    return out
