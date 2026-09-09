"""
risk_scorer.py

Risk scoring functions that fuse outputs from Dev 2 (ML anomaly scores),
the GNN (graph classification confidence), and Dev 1 (MITRE severity weights).

Formula:
    raw = 0.4 × composite_ml_score
        + 0.4 × gnn_confidence
        + 0.2 × mitre_severity_weight
    risk_score = min(100, int(raw × 100))

Severity thresholds:
    critical : risk_score >= 80
    high     : risk_score >= 60
    medium   : risk_score >= 40
    low      : risk_score <  40
"""

import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Severity weight defaults per MITRE technique — used when Dev 1's matched
# rules don't carry an explicit severity_weight field.
# Values are in [0, 1]; 1.0 = maximum severity contribution.
# ---------------------------------------------------------------------------
TECHNIQUE_DEFAULT_SEVERITY: dict[str, float] = {
    "T1190": 0.75,  # Exploit Public-Facing Application — high initial access risk
    "T1110": 0.55,  # Brute Force — moderate; common, often automated
    "T1059": 0.65,  # Command and Scripting Interpreter — significant execution risk
    "T1547": 0.70,  # Boot/Logon Autostart — persistence = elevated
    "T1490": 0.85,  # Inhibit System Recovery — direct precursor to ransomware
    "T1486": 1.00,  # Data Encrypted for Impact — maximum severity
    "T1048": 0.80,  # Exfiltration — data loss, critical
    "T1003": 0.75,  # OS Credential Dumping — enables lateral movement
}


def compute_risk_score(
    composite_ml_score: float,    # ANOMALY_SCORE_OBJECT.composite_ml_score  [0.0, 1.0]
    gnn_confidence: float,        # GNN_OUTPUT.gnn_confidence                [0.0, 1.0]
    mitre_severity_weight: float, # MITRE_RULES[n].severity_weight           [0.0, 1.0]
) -> int:
    """
    Compute an integer risk score in range [0, 100].

    Weights:
        0.3 — composite_ml_score (LSTM + IForest blend from Dev 2)
        0.1 — gnn_confidence     (node classification confidence from GNN)
        0.6 — mitre_severity_weight (technique-level severity from Dev 1 rules)

    All three inputs must be floats in [0.0, 1.0].
    Inputs outside that range are clamped with a warning.
    """
    # Defensive clamping — don't trust upstream to always be in range
    cms = _clamp(composite_ml_score, "composite_ml_score")
    gnn = _clamp(gnn_confidence, "gnn_confidence")
    msw = _clamp(mitre_severity_weight, "mitre_severity_weight")

    raw = (0.3 * cms) + (0.1 * gnn) + (0.6 * msw)
    # Ensure that a highly severe rule match guarantees a high risk score
    raw = max(raw, msw)
    score = min(100, int(raw * 100))

    logger.debug(
        "risk_score: ml=%.3f gnn=%.3f mitre_w=%.3f → raw=%.4f → score=%d",
        cms, gnn, msw, raw, score,
    )
    return score


def compute_severity(risk_score: int) -> str:
    """
    Map an integer risk_score [0, 100] to a severity string.

    Thresholds (inclusive lower bound):
        critical : >= 80
        high     : >= 60
        medium   : >= 40
        low      : <  40
    """
    if risk_score >= 80:
        return "critical"
    if risk_score >= 60:
        return "high"
    if risk_score >= 40:
        return "medium"
    return "low"


def aggregate_mitre_severity(matched_rules: list[dict]) -> float:
    """
    Derive a single mitre_severity_weight from a list of matched MITRE rules.
    Takes the maximum severity_weight across all matched rules (worst-case wins).

    Falls back to TECHNIQUE_DEFAULT_SEVERITY if the rule has no severity_weight.

    Args:
        matched_rules: list of MITRE_RULES entries from Dev 1

    Returns:
        float in [0.0, 1.0]
    """
    if not matched_rules:
        return 0.0

    weights = []
    for rule in matched_rules:
        if "severity_weight" in rule and rule["severity_weight"] is not None:
            weights.append(float(rule["severity_weight"]))
        else:
            tid = rule.get("technique_id", "")
            fallback = TECHNIQUE_DEFAULT_SEVERITY.get(tid, 0.5)
            logger.debug("No severity_weight for %s — using default %.2f", tid, fallback)
            weights.append(fallback)

    return max(weights)  # worst-case across matched techniques


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _clamp(value: float, name: str) -> float:
    if value < 0.0:
        logger.warning("%s=%.4f is below 0.0 — clamping to 0.0", name, value)
        return 0.0
    if value > 1.0:
        logger.warning("%s=%.4f is above 1.0 — clamping to 1.0", name, value)
        return 1.0
    return value
