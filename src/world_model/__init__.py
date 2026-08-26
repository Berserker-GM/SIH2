"""Traffic world-model data layer: CIC-IDS state windows S_t."""

from src.world_model.cic_schema import STATE_FEATURE_ORDER, INPUT_DIM
from src.world_model.labels import STAGE_NAMES, map_label
from src.world_model.model import LSTMWorldModel
from src.world_model.windows import WindowConfig, build_state_windows

__all__ = [
    "STATE_FEATURE_ORDER",
    "INPUT_DIM",
    "STAGE_NAMES",
    "map_label",
    "WindowConfig",
    "build_state_windows",
    "LSTMWorldModel",
]
