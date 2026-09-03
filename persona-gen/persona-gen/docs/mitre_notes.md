# MITRE ATT&CK-based persona reasoning

These personas have **zero real examples** in the CIC-IDS2018 training data — the
dataset structurally cannot tell you what recon/C2/exfiltration traffic looks like,
so parameters below are derived from ATT&CK technique descriptions instead of
dataset statistics. This is intentional and is the answer to 'how do you know this
isn't just made up' — it's traceable to a named technique, not a dataset fit.

## `recon_portscan_v1`

- **Family:** reconnaissance
- **Primary technique:** T1595.001
- **Adaptive:** False

T1595.001 (Active Scanning: Scanning IP Blocks) / T1046 (Network Service Discovery). No CIC-IDS2018 class covers pure port-sweep recon, so this is built from the technique description directly: sequential port selection over the well-known-port range, minimal-to-no payload (SYN probes only), near-zero bidirectionality (most probes get no meaningful response), and elevated RST (closed-port resets). 15% window-level blend toward a benign profile keeps this class from being trivially separable purely on port_scan_score.

## `recon_service_discovery_v1`

- **Family:** reconnaissance
- **Primary technique:** T1046
- **Adaptive:** False

T1046 (Network Service Discovery) — slower, more targeted than a raw port sweep: probes a curated list of commonly-exploited service ports rather than a sequential block, with longer inter-probe gaps ('longer inter-probe intervals to evade flow-threshold detection' per the brief) and slightly more of the probes getting a real service response (higher bidirectional_ratio than pure sweeping).

## `c2_beacon_v1`

- **Family:** command_and_control
- **Primary technique:** T1071.001
- **Adaptive:** False

T1071.001 (Application Layer Protocol: Web Protocols) — periodic low-volume beaconing over HTTPS/HTTP that blends with normal web traffic on the wire. Low flow rate (one beacon roughly every ~20s per window-lambda), small near-uniform payload sizes (check-in requests), timing jitter present but bounded — real C2 frameworks jitter beacons to avoid a perfectly periodic IAT signature, but not so much that they miss check-ins.

## `c2_dns_tunnel_v1`

- **Family:** command_and_control
- **Primary technique:** T1071.004
- **Adaptive:** False

T1071.004 (Application Layer Protocol: DNS) — C2 tunneled over DNS queries. Almost entirely UDP/port-53, moderate query-response bidirectionality, payload sizes larger than a normal DNS lookup (encoded C2 data in subdomain labels) but still small in absolute terms to stay under typical UDP fragmentation thresholds.

## `exfil_c2channel_v1`

- **Family:** exfiltration
- **Primary technique:** T1041
- **Adaptive:** False

T1041 (Exfiltration Over C2 Channel) — sustained, strongly one-directional (uplink-dominant, down_up_ratio_mean ~0.08 i.e. inverted from normal browsing) byte transfer over the same channel used for C2, held on a single port to look like one long-lived legitimate HTTPS session rather than many short flows.

## `evasive_recon_v1`

- **Family:** reconnaissance
- **Primary technique:** T1595.001
- **Adaptive:** True

Evasive variant of the port-sweep persona: hill-climbing tunes probe rate, jitter, and how aggressively port diversity drives port_scan_score, while the dual objective requires it still reach the 'confirm' phase within a 40-window budget — i.e. it must actually keep scanning, just less conspicuously.

## `evasive_c2_v1`

- **Family:** command_and_control
- **Primary technique:** T1071.001
- **Adaptive:** True

Evasive beaconing: hill-climbing widens beacon-interval jitter and payload-size variance to break up the periodic-beacon signature detectors look for, but the budget constraint means it can't just go silent — it must still reach 'sustain' (i.e. keep beaconing) within 50 windows.

## `evasive_lateral_v1`

- **Family:** lateral_movement
- **Primary technique:** T1021.002
- **Adaptive:** True

The named example in the brief: evasive SMB-style lateral movement. Hill-climbing tunes movement rate and port-diversity signal, but the persona must still complete the 'stage -> move' kill chain within 35 windows — an evasive persona that just goes quiet forever is not demonstrating evasion.
