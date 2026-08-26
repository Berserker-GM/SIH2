"""
isolation_forest.py — Dev 2 (Omkar)
Isolation Forest for point-level anomaly detection.

Detects outlier individual events that the LSTM may miss (e.g. single-event
bytes_sent spikes). Operates on the 9-feature subset that excludes the
normalised port fields (those carry less per-event signal for iForest).

Feature order MUST match Dev 1 feature_extractor.py FEATURE_ORDER exactly.
"""

import pickle
import numpy as np
from pathlib import Path
from sklearn.ensemble import IsolationForest

# ---------------------------------------------------------------------------
# Feature contract — sync with Dev 1 and lstm_autoencoder.py
# ---------------------------------------------------------------------------
FEATURE_ORDER = [
    "login_fail_count",
    "login_success_count",
    "unique_dest_ips",
    "bytes_sent_total",
    "file_ops_count",
    "cpu_pct_avg",
    "payload_flag_count",
    "inter_arrival_ms_avg",
    "entropy_dest_ports",
]
# !! MUST match Dev 1 feature_extractor.py FEATURE_ORDER exactly !!
# (src_port and dest_port are used by the LSTM but excluded from iForest
#  to avoid dimensionality noise on single-event scoring.)

MODEL_PATH       = Path(__file__).parent / "models" / "iforest.pkl"
BENIGN_BOOTSTRAP = 500   # events consumed during benign simulator phase


# ---------------------------------------------------------------------------
# Singleton scorer
# ---------------------------------------------------------------------------
class IForestScorer:
    def __init__(self):
        self.model = IsolationForest(
            n_estimators=200,
            contamination=0.05,   # assume ~5 % anomaly rate in live traffic
            random_state=42,
            n_jobs=-1,
        )
        self._fitted  = False
        self._buffer: list[list[float]] = []   # accumulates benign bootstrap rows
        self._score_min: float = -0.5
        self._score_max: float = 0.5

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: Path = MODEL_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"[IForest] Saved to {path}")

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> "IForestScorer":
        if path.exists():
            with open(path, "rb") as f:
                obj = pickle.load(f)
            print(f"[IForest] Loaded from {path}")
            return obj
        print(f"[IForest] No saved model at {path}; starting fresh.")
        return cls()

    # ------------------------------------------------------------------
    # Bootstrap fitting during simulator benign phase
    # ------------------------------------------------------------------
    def add_benign_event(self, feature_vector: dict) -> bool:
        """
        Buffer an event from the benign simulator phase.
        Auto-fits once BENIGN_BOOTSTRAP events are collected.

        Returns True when the model has just been fitted.
        """
        row = self._fv_to_row(feature_vector)
        self._buffer.append(row)

        if len(self._buffer) >= BENIGN_BOOTSTRAP and not self._fitted:
            X = np.array(self._buffer, dtype=np.float32)
            self.model.fit(X)
            self._fitted = True
            print(f"[IForest] Auto-fitted on {len(self._buffer)} benign events.")
            self._buffer.clear()
            return True

        return False

    def fit(self, feature_vectors: list[dict]) -> None:
        """Explicit fit from a list of ML_FEATURE_VECTOR dicts (e.g. from CSV)."""
        X = np.array([self._fv_to_row(fv) for fv in feature_vectors], dtype=np.float32)
        self.model.fit(X)
        self._fitted = True
        print(f"[IForest] Fitted on {len(X)} samples.")

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------
    def score(self, feature_vector: dict) -> float:
        """
        Score a single ML_FEATURE_VECTOR dict.

        Returns:
            iforest_score (float): normalised anomaly score in [0.0, 1.0].
                                   Higher → more anomalous.
        """
        if not self._fitted:
            # Return neutral score until fitted — anomaly_scorer handles this
            return 0.5

        row = np.array(self._fv_to_row(feature_vector), dtype=np.float32).reshape(1, -1)
        # decision_function returns negative = anomaly, positive = normal
        raw = float(self.model.decision_function(row)[0])

        # Online min-max normalisation (inverted so high score = more anomalous)
        self._score_min = min(self._score_min, raw)
        self._score_max = max(self._score_max, raw)
        denom = max(self._score_max - self._score_min, 1e-8)
        normalised = float(np.clip((raw - self._score_min) / denom, 0.0, 1.0))
        # Invert: high raw_score = normal → we want high iforest_score = anomalous
        return round(1.0 - normalised, 6)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _fv_to_row(fv: dict) -> list[float]:
        return [float(fv.get(k, 0.0)) for k in FEATURE_ORDER]


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
_scorer: IForestScorer | None = None


def get_scorer() -> IForestScorer:
    """Return the module-level IForestScorer singleton."""
    global _scorer
    if _scorer is None:
        _scorer = IForestScorer.load(MODEL_PATH)
    return _scorer
