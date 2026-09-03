"""Builds docs/mitre_notes.md from the _notes field of each MITRE-doc-based
and adaptive persona config — a single place to point a jury/judge instead
of reconstructing the reasoning on the spot."""
import json
from pathlib import Path

TARGET_PERSONAS = [
    "recon_portscan_v1", "recon_service_discovery_v1",
    "c2_beacon_v1", "c2_dns_tunnel_v1",
    "exfil_c2channel_v1",
    "evasive_recon_v1", "evasive_c2_v1", "evasive_lateral_v1",
]

lines = [
    "# MITRE ATT&CK-based persona reasoning",
    "",
    "These personas have **zero real examples** in the CIC-IDS2018 training data — the",
    "dataset structurally cannot tell you what recon/C2/exfiltration traffic looks like,",
    "so parameters below are derived from ATT&CK technique descriptions instead of",
    "dataset statistics. This is intentional and is the answer to 'how do you know this",
    "isn't just made up' — it's traceable to a named technique, not a dataset fit.",
    "",
]

for pid in TARGET_PERSONAS:
    with open(f"personas/configs/{pid}.json") as f:
        cfg = json.load(f)
    lines.append(f"## `{pid}`")
    lines.append("")
    lines.append(f"- **Family:** {cfg['family']}")
    lines.append(f"- **Primary technique:** {cfg.get('mitre_technique') or '(see phases)'}")
    lines.append(f"- **Adaptive:** {cfg['adaptive']}")
    if cfg.get("_notes"):
        lines.append("")
        lines.append(cfg["_notes"])
    lines.append("")

Path("docs").mkdir(exist_ok=True)
Path("docs/mitre_notes.md").write_text("\n".join(lines))
print("wrote docs/mitre_notes.md")
