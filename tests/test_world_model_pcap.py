#!/usr/bin/env python3
"""PCAP-only packet stats (TTL / fragment / retransmit) — sidecar, not LSTM S_t."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.world_model.cic_schema import (
    PACKET_LEVEL_SPEC,
    PACKET_LEVEL_INDEX,
    PACKET_LEVEL_UNIT,
    STATE_FEATURE_ORDER,
)
from src.world_model.pcap_features import packet_window_stats, stats_for_pcap_window

results = []


def check(name: str, condition: bool, detail: str = "") -> None:
    results.append((name, condition))
    icon = "PASS" if condition else "FAIL"
    extra = f"  ({detail})" if detail else ""
    print(f"  {icon}  {name}{extra}")


print("\n-- Packet-level dims 22–31 (names, units, order) ----------------------")
names = [row["name"] for row in PACKET_LEVEL_SPEC]
indices = [row["index"] for row in PACKET_LEVEL_SPEC]
check("ten packet-level rows", len(PACKET_LEVEL_SPEC) == 10)
check("indices 22..31", indices == list(range(22, 32)))
check("names match STATE_FEATURE_ORDER[22:]", names == list(STATE_FEATURE_ORDER[22:]))
check("pkt_len_mean unit bytes", PACKET_LEVEL_UNIT["pkt_len_mean"] == "bytes")
check("pkt_len_std unit bytes", PACKET_LEVEL_UNIT["pkt_len_std"] == "bytes")
check("pkt_len_max unit bytes", PACKET_LEVEL_UNIT["pkt_len_max"] == "bytes")
check("init_fwd_win_mean unit bytes", PACKET_LEVEL_UNIT["init_fwd_win_mean"] == "bytes")
check("init_bwd_win_mean unit bytes", PACKET_LEVEL_UNIT["init_bwd_win_mean"] == "bytes")
check("fwd_psh_flags_sum unit count", PACKET_LEVEL_UNIT["fwd_psh_flags_sum"] == "count")
check("port_scan_score unit [0,1]", PACKET_LEVEL_UNIT["port_scan_score"] == "score [0, 1]")
check("ttl_variance is TTL²", "TTL²" in PACKET_LEVEL_UNIT["ttl_variance"])
check("ip_fragment_flags unit packets", PACKET_LEVEL_UNIT["ip_fragment_flags"] == "packets")
check("retransmit_count unit packets", PACKET_LEVEL_UNIT["retransmit_count"] == "packets")
check("ttl_variance index 29", PACKET_LEVEL_INDEX["ttl_variance"] == 29)
check("sources: cic then derived then pcap", [row["source"] for row in PACKET_LEVEL_SPEC] == [
    "cic", "cic", "cic", "cic", "cic", "cic", "derived", "pcap", "pcap", "pcap",
])


print("\n-- Synthetic PCAP window stats ----------------------------------------")
from scapy.layers.inet import IP, TCP  # noqa: E402
from scapy.layers.l2 import Ether  # noqa: E402
from scapy.packet import Raw  # noqa: E402
from scapy.plist import PacketList  # noqa: E402

pkts = PacketList()
pkts.append(
    Ether()
    / IP(src="10.0.0.1", dst="10.0.0.2", ttl=64)
    / TCP(sport=1234, dport=80, seq=100, flags="A")
    / Raw(b"abc")
)
pkts.append(
    Ether()
    / IP(src="10.0.0.1", dst="10.0.0.2", ttl=128)
    / TCP(sport=1234, dport=80, seq=100, flags="A")
    / Raw(b"abc")
)
big = (
    Ether()
    / IP(src="10.0.0.3", dst="10.0.0.4", ttl=64)
    / TCP(sport=1, dport=2, seq=1, flags="S")
    / Raw(b"X" * 2000)
)
from scapy.layers.inet import fragment  # noqa: E402

pkts.extend(fragment(big, fragsize=1000))

stats = packet_window_stats(pkts)
check("ttl variance > 0", stats["ttl_variance"] > 0.0, str(stats["ttl_variance"]))
check("fragment flags counted", stats["ip_fragment_flags"] >= 1.0, str(stats["ip_fragment_flags"]))
check("retransmit counted", stats["retransmit_count"] >= 1.0, str(stats["retransmit_count"]))
check("does not change 32-d layout", len(STATE_FEATURE_ORDER) == 32)
check(
    "pcap names still stubs in S_t",
    list(STATE_FEATURE_ORDER[-3:]) == ["ttl_variance", "ip_fragment_flags", "retransmit_count"],
)

empty = packet_window_stats([])
check("empty window zeros", empty["ttl_variance"] == 0.0 and empty["retransmit_count"] == 0.0)


print("\n-- Scapy PcapReader ingest for last demo window -----------------------")
from datetime import datetime, timezone
from scapy.utils import wrpcap

t0 = datetime(2018, 2, 28, 10, 1, 50, tzinfo=timezone.utc).timestamp()
timed = []
for i, pkt in enumerate(pkts):
    pkt.time = t0 + 0.05 * i
    timed.append(pkt)
pcap_path = PROJECT_ROOT / "data" / "processed" / "_test_sidecar.pcap"
pcap_path.parent.mkdir(parents=True, exist_ok=True)
wrpcap(str(pcap_path), timed)
file_stats = stats_for_pcap_window(pcap_path, t0, t0 + 5.0)
check("file ingest ttl var > 0", file_stats["ttl_variance"] > 0.0, str(file_stats["ttl_variance"]))
check("file ingest fragments", file_stats["ip_fragment_flags"] >= 1.0)
check("file ingest retransmit", file_stats["retransmit_count"] >= 1.0)
check("sidecar not injected", file_stats["injected_into_st"] is False)
outside = stats_for_pcap_window(pcap_path, t0 + 60.0, t0 + 65.0)
check("non-overlapping window is empty", outside["n_packets"] == 0.0)
pcap_path.unlink(missing_ok=True)


print("\n-- Summary ------------------------------------------------------------")
passed = sum(1 for _, ok in results if ok)
failed = sum(1 for _, ok in results if not ok)
print(f"\n  Total: {len(results)}   Passed: {passed}   Failed: {failed}\n")
if failed:
    sys.exit(1)
print("  PCAP sidecar stats are ready (not injected into frozen S_t).\n")
