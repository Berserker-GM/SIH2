"""
mitre_transitions.py 

Hardcoded MITRE ATT&CK next-step transition matrix for the demo.
Encodes empirically likely attack progressions observed in real-world campaigns.

TRANSITION_MATRIX[current_technique] = [(next_technique_id, base_probability), ...]
All probability lists per technique sum to 1.0.

Techniques covered (7):
  T1190 — Exploit Public-Facing Application
  T1110 — Brute Force
  T1059 — Command and Scripting Interpreter
  T1547 — Boot or Logon Autostart Execution
  T1490 — Inhibit System Recovery
  T1486 — Data Encrypted for Impact (ransomware payload)
  T1048 — Exfiltration Over Alternative Protocol
  T1003 — OS Credential Dumping  (terminal node — no outgoing transitions defined)
"""

# ---------------------------------------------------------------------------
# Technique metadata
# ---------------------------------------------------------------------------

TECHNIQUE_NAMES: dict[str, str] = {
    "T1190": "Exploit Public-Facing Application",
    "T1110": "Brute Force",
    "T1059": "Command and Scripting Interpreter",
    "T1547": "Boot or Logon Autostart Execution",
    "T1490": "Inhibit System Recovery",
    "T1486": "Data Encrypted for Impact",
    "T1048": "Exfiltration Over Alternative Protocol",
    "T1003": "OS Credential Dumping",
}

# MITRE tactic mapping — technique_id → tactic name
# Used by incident_assembler.py when building attack_chain and predicted_next
TECHNIQUE_TACTICS: dict[str, str] = {
    "T1190": "Initial Access",
    "T1110": "Credential Access",
    "T1059": "Execution",
    "T1547": "Persistence",
    "T1490": "Impact",
    "T1486": "Impact",
    "T1048": "Exfiltration",
    "T1003": "Credential Access",
}

# ---------------------------------------------------------------------------
# Transition matrix
# Probabilities per row sum to 1.0.
# Represents: given the attacker just executed technique X,
# what technique do they most likely execute next?
# ---------------------------------------------------------------------------

TRANSITION_MATRIX: dict[str, list[tuple[str, float]]] = {
    # Initial exploit → command execution (most common), persistence, or exfiltration
    "T1190": [
        ("T1059", 0.72),
        ("T1547", 0.18),
        ("T1048", 0.10),
    ],

    # Brute force → command execution, back to exploitation, or persistence
    "T1110": [
        ("T1059", 0.45),
        ("T1190", 0.35),
        ("T1547", 0.20),
    ],

    # Command execution → persistence, disabling recovery, or credential dumping
    "T1059": [
        ("T1547", 0.40),
        ("T1490", 0.35),
        ("T1003", 0.25),
    ],

    # Persistence → ransomware payload, exfiltration, or credential dumping
    "T1547": [
        ("T1486", 0.50),
        ("T1048", 0.30),
        ("T1003", 0.20),
    ],

    # Inhibit system recovery → almost certainly ransomware, sometimes exfil
    "T1490": [
        ("T1486", 0.80),
        ("T1048", 0.20),
    ],

    # Ransomware payload → exfiltration (double extortion), or inhibit recovery loop
    "T1486": [
        ("T1048", 0.90),
        ("T1490", 0.10),
    ],

    # Exfiltration → loop back to persistence or credential dumping
    "T1048": [
        ("T1547", 0.60),
        ("T1003", 0.40),
    ],

    # T1003 (OS Credential Dumping) is a terminal node in this graph:
    # no defined outgoing transitions — attacker has achieved their objective.
    # incident_assembler handles the None case gracefully.
    "T1003": [],
}


# ---------------------------------------------------------------------------
# Helper — get the most probable next technique given the current one
# ---------------------------------------------------------------------------

def get_top_next_technique(current_technique: str) -> tuple[str, float] | None:
    """
    Returns (technique_id, probability) of the most likely next step,
    or None if current_technique is a terminal node or unknown.
    """
    transitions = TRANSITION_MATRIX.get(current_technique, [])
    if not transitions:
        return None
    # Already sorted descending by probability in the matrix definition,
    # but be defensive:
    return max(transitions, key=lambda t: t[1])


def get_transition_probs(current_technique: str) -> dict[str, float]:
    """
    Returns {technique_id: probability} dict for all possible next steps.
    Returns empty dict for terminal nodes.
    """
    return dict(TRANSITION_MATRIX.get(current_technique, []))
