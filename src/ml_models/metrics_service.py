"""
schema_validator.py — Dev 2 (Omkar)
Validates dicts against the shared integration contract JSON schemas.
All four devs can import and use this to catch contract violations early.

Usage:
    from schema_validator import validate_normalized_event, validate_anomaly_score

    ok, err = validate_normalized_event(event_dict)
    ok, err = validate_anomaly_score(score_dict)
    ok, err = validate_ml_feature_vector(fv_dict)
"""

import json
import logging
from pathlib import Path

log = logging.getLogger("schema_validator")

# ---------------------------------------------------------------------------
# Locate contracts directory (project root / contracts)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]   # goes up to Normnative/
CONTRACTS_DIR = PROJECT_ROOT / "contracts"

# ---------------------------------------------------------------------------
# Define schema variables (initialise to None)
# ---------------------------------------------------------------------------
_NORMALIZED_EVENT_SCHEMA = None
_ANOMALY_SCORE_SCHEMA = None
_ML_FEATURE_VECTOR_SCHEMA = None
_SCHEMAS_LOADED = False


def _load(filename: str) -> dict | None:
    """Load a JSON schema from the contracts directory."""
    path = CONTRACTS_DIR / filename
    if not path.exists():
        log.warning("Schema file not found: %s", path)
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        log.error("Failed to load schema %s: %s", path, e)
        return None


# Try to load all schemas
_NORMALIZED_EVENT_SCHEMA = _load("normalized_event_object.schema.json")
_ANOMALY_SCORE_SCHEMA = _load("anomaly_score_object.schema.json")
_ML_FEATURE_VECTOR_SCHEMA = _load("ml_feature_vector.schema.json")

if all(s is not None for s in (_NORMALIZED_EVENT_SCHEMA, _ANOMALY_SCORE_SCHEMA, _ML_FEATURE_VECTOR_SCHEMA)):
    _SCHEMAS_LOADED = True
    log.info("All contract schemas loaded successfully from %s", CONTRACTS_DIR)
else:
    log.warning("Some contract schemas missing – validation will be skipped.")


# ---------------------------------------------------------------------------
# Core validator (uses jsonschema if available, falls back to key checks)
# ---------------------------------------------------------------------------
def _validate(instance: dict, schema: dict | None) -> tuple[bool, str]:
    """
    Returns (True, "") on success or (False, error_message) on failure.
    If schema is None or jsonschema not installed, does a lightweight check.
    """
    if schema is None:
        # No schema → skip validation
        return True, ""

    try:
        import jsonschema
        jsonschema.validate(instance=instance, schema=schema)
        return True, ""
    except ImportError:
        # jsonschema not installed — do a lightweight required-field check
        required = schema.get("required", [])
        missing = [k for k in required if k not in instance]
        if missing:
            return False, f"Missing required fields: {missing}"
        # Check no extra fields if additionalProperties=false
        if schema.get("additionalProperties") is False:
            extra = [k for k in instance if k not in schema.get("properties", {})]
            if extra:
                return False, f"Extra fields not allowed: {extra}"
        return True, ""
    except jsonschema.ValidationError as exc:
        return False, exc.message
    except Exception as exc:
        return False, str(exc)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def validate_normalized_event(event: dict) -> tuple[bool, str]:
    """Validate a NormalizedEventObject dict (Dev 1 output)."""
    return _validate(event, _NORMALIZED_EVENT_SCHEMA)


def validate_anomaly_score(score: dict) -> tuple[bool, str]:
    """Validate an AnomalyScoreObject dict (Dev 2 output)."""
    return _validate(score, _ANOMALY_SCORE_SCHEMA)


def validate_ml_feature_vector(fv: dict) -> tuple[bool, str]:
    """Validate an MLFeatureVector dict (Dev 1 → Dev 2 intermediate)."""
    return _validate(fv, _ML_FEATURE_VECTOR_SCHEMA)


def assert_normalized_event(event: dict) -> None:
    """Raise ValueError if the event dict violates the contract."""
    ok, err = validate_normalized_event(event)
    if not ok:
        raise ValueError(f"NormalizedEventObject contract violation: {err}")


def assert_anomaly_score(score: dict) -> None:
    """Raise ValueError if the score dict violates the contract."""
    ok, err = validate_anomaly_score(score)
    if not ok:
        raise ValueError(f"AnomalyScoreObject contract violation: {err}")


# ---------------------------------------------------------------------------
# CLI: validate a JSON file against a named schema
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python schema_validator.py <schema_name> <json_file>")
        print("  schema_name: normalized_event | anomaly_score | ml_feature_vector")
        sys.exit(1)

    schema_name, json_file = sys.argv[1], sys.argv[2]
    with open(json_file) as f:
        data = json.load(f)

    validators = {
        "normalized_event":   validate_normalized_event,
        "anomaly_score":      validate_anomaly_score,
        "ml_feature_vector":  validate_ml_feature_vector,
    }
    if schema_name not in validators:
        print(f"Unknown schema '{schema_name}'. Choose from: {list(validators)}")
        sys.exit(1)

    ok, err = validators[schema_name](data)
    if ok:
        print(f"✅  {json_file} is valid against '{schema_name}' schema.")
    else:
        print(f"❌  Validation failed: {err}")
        sys.exit(1)