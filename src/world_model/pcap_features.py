"""PCAP-only packet features (SIH Section 1).

CIC ML CSVs do not have TTL, IP fragment flags, or retransmit counts.
Those three coordinates of S_t stay 0 for CSV-only inference so the frozen
LSTM/scaler are not fed out-of-distribution values.

When a PCAP is available, this module computes the same three stats for the
matching 5-second window as a sidecar (pcap_stats). Filling them into S_t
requires a retrain.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np

PCAP_STAT_NAMES = ("ttl_variance", "ip_fragment_flags", "retransmit_count")


def _ip_layer(pkt: Any):
    if pkt.haslayer("IP"):
        return pkt["IP"]
    if pkt.haslayer("IPv6"):
        return pkt["IPv6"]
    return None


def packet_window_stats(packets: Iterable[Any]) -> dict[str, float]:
    """Aggregate TTL variance, fragment flags, TCP retransmits over one window."""
    ttls: list[float] = []
    fragment_flags = 0.0
    seq_seen: Counter[tuple] = Counter()
    retransmits = 0.0

    for pkt in packets:
        ip = _ip_layer(pkt)
        if ip is None:
            continue
        ttl = getattr(ip, "ttl", None)
        if ttl is None:
            ttl = getattr(ip, "hlim", None)
        if ttl is not None:
            ttls.append(float(ttl))

        frag = int(getattr(ip, "frag", 0) or 0)
        flags = getattr(ip, "flags", 0)
        mf = bool(int(flags) & 0x1) if flags is not None else False
        if frag > 0 or mf:
            fragment_flags += 1.0

        if pkt.haslayer("TCP") and hasattr(pkt, "__getitem__"):
            tcp = pkt["TCP"]
            payload_len = int(len(bytes(tcp.payload))) if tcp.payload else 0
            if payload_len <= 0:
                continue
            key = (
                str(getattr(ip, "src", "")),
                str(getattr(ip, "dst", "")),
                int(tcp.sport),
                int(tcp.dport),
                int(tcp.seq),
            )
            seq_seen[key] += 1
            if seq_seen[key] > 1:
                retransmits += 1.0

    ttl_var = float(np.var(np.asarray(ttls, dtype=np.float64))) if len(ttls) >= 2 else 0.0
    return {
        "ttl_variance": ttl_var,
        "ip_fragment_flags": float(fragment_flags),
        "retransmit_count": float(retransmits),
        "n_packets": float(len(ttls)),
    }


def stats_for_pcap_window(
    path: str | Path,
    t0: float,
    t1: float,
    *,
    assume_sorted: bool = True,
) -> dict[str, float]:
    """Read a PCAP with Scapy and aggregate packet-level dims 29–31 for [t0, t1)."""
    from scapy.utils import PcapReader

    pcap = Path(path)
    if not pcap.is_file():
        raise ValueError(f"PCAP not found: {pcap}")
    selected: list[Any] = []
    with PcapReader(str(pcap)) as reader:
        for pkt in reader:
            ts = float(pkt.time)
            if ts < t0:
                continue
            if ts >= t1:
                if assume_sorted:
                    break
                continue
            selected.append(pkt)
    out = packet_window_stats(selected)
    out["t0"] = float(t0)
    out["t1"] = float(t1)
    out["injected_into_st"] = False
    out["source_file"] = pcap.name
    return out
