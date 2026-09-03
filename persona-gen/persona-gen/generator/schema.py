"""
Shared interface contract constants.

This file must stay byte-identical in spirit to the pipeline dev's copy —
if the CSV schema or stage list ever needs to change, that's a cross-team
conversation, not a local edit.
"""

MITRE_STAGES = [
    "benign",
    "reconnaissance",
    "initial_access",
    "lateral_movement",
    "command_and_control",
    "exfiltration",
    "impact",
]

WINDOW_SECONDS = 5

METADATA_COLUMNS = ["timestamp", "persona_id", "run_id", "mitre_stage", "technique_id"]

# Exact order required by the shared contract. Do not reorder.
FEATURE_COLUMNS = [
    "flow_count", "unique_dst_ports", "dest_port_entropy", "tcp_ratio", "udp_ratio",
    "bytes_fwd_sum", "bytes_bwd_sum", "bidirectional_ratio", "pkts_fwd_sum", "pkts_bwd_sum",
    "duration_mean", "duration_max", "iat_mean", "iat_std", "iat_max",
    "syn_flag_sum", "ack_flag_sum", "fin_flag_sum", "rst_flag_sum", "psh_flag_sum", "urg_flag_sum",
    "down_up_ratio_mean", "pkt_len_mean", "pkt_len_std", "pkt_len_max",
    "init_fwd_win_mean", "init_bwd_win_mean", "fwd_psh_flags_sum", "port_scan_score",
    "ttl_variance", "ip_fragment_flags", "retransmit_count",
]

assert len(FEATURE_COLUMNS) == 32, f"contract requires 32 feature columns, got {len(FEATURE_COLUMNS)}"

CSV_COLUMNS = METADATA_COLUMNS + FEATURE_COLUMNS

SCORE_HISTORY_WINDOWS = 8  # x: shape (8, 32) per the pipeline dev's score_history() contract
