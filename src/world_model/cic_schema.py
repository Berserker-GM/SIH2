"""
CIC-IDS2017 / CSE-CIC-IDS2018 column aliases → SIH network-state features.

CIC Machine-Learning CSVs do not include Src IP / Dst IP / Src Port.
Those fields are reserved (0) until PCAP or GeneratedLabelledFlows are used.
"""

from __future__ import annotations

# Canonical CIC-IDS2018 ML CSV header (80 columns), confirmed from
# s3://cse-cic-ids2018/Processed Traffic Data for ML Algorithms/
CIC2018_COLUMNS = [
    "Dst Port", "Protocol", "Timestamp", "Flow Duration",
    "Tot Fwd Pkts", "Tot Bwd Pkts", "TotLen Fwd Pkts", "TotLen Bwd Pkts",
    "Fwd Pkt Len Max", "Fwd Pkt Len Min", "Fwd Pkt Len Mean", "Fwd Pkt Len Std",
    "Bwd Pkt Len Max", "Bwd Pkt Len Min", "Bwd Pkt Len Mean", "Bwd Pkt Len Std",
    "Flow Byts/s", "Flow Pkts/s",
    "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
    "Fwd IAT Tot", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min",
    "Bwd IAT Tot", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min",
    "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags",
    "Fwd Header Len", "Bwd Header Len", "Fwd Pkts/s", "Bwd Pkts/s",
    "Pkt Len Min", "Pkt Len Max", "Pkt Len Mean", "Pkt Len Std", "Pkt Len Var",
    "FIN Flag Cnt", "SYN Flag Cnt", "RST Flag Cnt", "PSH Flag Cnt",
    "ACK Flag Cnt", "URG Flag Cnt", "CWE Flag Count", "ECE Flag Cnt",
    "Down/Up Ratio", "Pkt Size Avg", "Fwd Seg Size Avg", "Bwd Seg Size Avg",
    "Fwd Byts/b Avg", "Fwd Pkts/b Avg", "Fwd Blk Rate Avg",
    "Bwd Byts/b Avg", "Bwd Pkts/b Avg", "Bwd Blk Rate Avg",
    "Subflow Fwd Pkts", "Subflow Fwd Byts", "Subflow Bwd Pkts", "Subflow Bwd Byts",
    "Init Fwd Win Byts", "Init Bwd Win Byts", "Fwd Act Data Pkts", "Fwd Seg Size Min",
    "Active Mean", "Active Std", "Active Max", "Active Min",
    "Idle Mean", "Idle Std", "Idle Max", "Idle Min",
    "Label",
]

# Each key is a logical field. Values are accepted CSV header variants
# (after strip + casefold) from CIC-IDS2017 and CSE-CIC-IDS2018.
COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "dst_port": ("dst port", "destination port", "destinationport", "dstport"),
    "src_port": ("src port", "source port", "sourceport", "srcport"),
    "src_ip": ("src ip", "source ip", "sourceip", "srcip"),
    "dst_ip": ("dst ip", "destination ip", "destinationip", "dstip"),
    "protocol": ("protocol",),
    "timestamp": ("timestamp", "flow start", "flowstart"),
    "flow_duration": ("flow duration", "flowduration"),
    "tot_fwd_pkts": ("tot fwd pkts", "total fwd packets", "totalfwdpackets"),
    "tot_bwd_pkts": ("tot bwd pkts", "total backward packets", "totalbwdpackets"),
    "totlen_fwd": ("totlen fwd pkts", "total length of fwd packets"),
    "totlen_bwd": ("totlen bwd pkts", "total length of bwd packets"),
    "flow_iat_mean": ("flow iat mean",),
    "flow_iat_std": ("flow iat std",),
    "flow_iat_max": ("flow iat max",),
    "syn_flag": ("syn flag cnt", "syn flag count"),
    "ack_flag": ("ack flag cnt", "ack flag count"),
    "fin_flag": ("fin flag cnt", "fin flag count"),
    "rst_flag": ("rst flag cnt", "rst flag count"),
    "psh_flag": ("psh flag cnt", "psh flag count"),
    "urg_flag": ("urg flag cnt", "urg flag count"),
    "fwd_psh": ("fwd psh flags",),
    "down_up_ratio": ("down/up ratio", "down / up ratio"),
    "pkt_len_mean": ("pkt len mean", "packet length mean"),
    "pkt_len_std": ("pkt len std", "packet length std"),
    "pkt_len_max": ("pkt len max", "packet length max", "max packet length"),
    "pkt_len_min": ("pkt len min", "packet length min", "min packet length"),
    "init_fwd_win": ("init fwd win byts", "init_win_bytes_forward"),
    "init_bwd_win": ("init bwd win byts", "init_win_bytes_backward"),
    "label": ("label", "class"),
}

# Ordered state vector S_t. Indices 0–21 are flow-level; 22–31 packet-level.
# Indices 29–31 stay 0 on CIC CSVs until a PCAP extractor fills them.
STATE_FEATURE_ORDER = [
    # --- flow-level (NetFlow / IPFIX-style aggregates) ---
    "flow_count",
    "unique_dst_ports",
    "dest_port_entropy",
    "tcp_ratio",
    "udp_ratio",
    "bytes_fwd_sum",
    "bytes_bwd_sum",
    "bidirectional_ratio",
    "pkts_fwd_sum",
    "pkts_bwd_sum",
    "duration_mean",
    "duration_max",
    "iat_mean",
    "iat_std",
    "iat_max",
    "syn_flag_sum",
    "ack_flag_sum",
    "fin_flag_sum",
    "rst_flag_sum",
    "psh_flag_sum",
    "urg_flag_sum",
    "down_up_ratio_mean",
    # --- packet-level (CICFlowMeter packet stats + PCAP stubs) ---
    "pkt_len_mean",
    "pkt_len_std",
    "pkt_len_max",
    "init_fwd_win_mean",
    "init_bwd_win_mean",
    "fwd_psh_flags_sum",
    "port_scan_score",
    "ttl_variance",          # PCAP-only
    "ip_fragment_flags",     # PCAP-only
    "retransmit_count",      # PCAP-only
]

INPUT_DIM = len(STATE_FEATURE_ORDER)  # 32

PCAP_ONLY_FEATURES = ("ttl_variance", "ip_fragment_flags", "retransmit_count")

# Packet-level slice of S_t (indices 22–31). Units are what windows.py writes
# into the unscaled vector (LSTM then z-scores with the train scaler).
# CIC-IDS2018 duration / IAT columns are microseconds; packet lengths and
# TCP windows are bytes.
PACKET_LEVEL_SPEC = (
    {
        "index": 22,
        "name": "pkt_len_mean",
        "unit": "bytes",
        "cic_column": "Pkt Len Mean",
        "agg": "mean of per-flow packet-length means",
        "source": "cic",
    },
    {
        "index": 23,
        "name": "pkt_len_std",
        "unit": "bytes",
        "cic_column": "Pkt Len Std",
        "agg": "mean of per-flow packet-length stds",
        "source": "cic",
    },
    {
        "index": 24,
        "name": "pkt_len_max",
        "unit": "bytes",
        "cic_column": "Pkt Len Max",
        "agg": "max of per-flow packet-length maxima",
        "source": "cic",
    },
    {
        "index": 25,
        "name": "init_fwd_win_mean",
        "unit": "bytes",
        "cic_column": "Init Fwd Win Byts",
        "agg": "mean of initial forward TCP window",
        "source": "cic",
    },
    {
        "index": 26,
        "name": "init_bwd_win_mean",
        "unit": "bytes",
        "cic_column": "Init Bwd Win Byts",
        "agg": "mean of initial backward TCP window",
        "source": "cic",
    },
    {
        "index": 27,
        "name": "fwd_psh_flags_sum",
        "unit": "count",
        "cic_column": "Fwd PSH Flags",
        "agg": "sum of forward PSH flag counts",
        "source": "cic",
    },
    {
        "index": 28,
        "name": "port_scan_score",
        "unit": "score [0, 1]",
        "cic_column": None,
        "agg": "max(sequential dst-port fraction, spread×entropy)",
        "source": "derived",
    },
    {
        "index": 29,
        "name": "ttl_variance",
        "unit": "TTL² (hop-count variance, ddof=0)",
        "cic_column": None,
        "agg": "population variance of IP TTL / IPv6 Hop Limit in the 5s bin",
        "source": "pcap",
    },
    {
        "index": 30,
        "name": "ip_fragment_flags",
        "unit": "packets",
        "cic_column": None,
        "agg": "count of IPv4 packets with MF or fragment offset > 0",
        "source": "pcap",
    },
    {
        "index": 31,
        "name": "retransmit_count",
        "unit": "packets",
        "cic_column": None,
        "agg": "count of TCP segments with payload that repeat (src,dst,sport,dport,seq)",
        "source": "pcap",
    },
)

PACKET_LEVEL_INDEX = {row["name"]: row["index"] for row in PACKET_LEVEL_SPEC}
PACKET_LEVEL_UNIT = {row["name"]: row["unit"] for row in PACKET_LEVEL_SPEC}

assert tuple(row["name"] for row in PACKET_LEVEL_SPEC) == tuple(STATE_FEATURE_ORDER[22:])
assert [row["index"] for row in PACKET_LEVEL_SPEC] == list(range(22, 32))


def normalize_header(name: str) -> str:
    return " ".join(str(name).strip().replace("_", " ").split()).casefold()


def bind_columns(columns: list[str]) -> dict[str, str]:
    """Map logical field → actual dataframe column name."""
    lookup = {normalize_header(c): c for c in columns}
    bound: dict[str, str] = {}
    for field, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in lookup:
                bound[field] = lookup[alias]
                break
    return bound
