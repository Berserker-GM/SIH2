# conftest.py  — root level
# Adds all source roots to sys.path so every test and every module
# can import without relative path hacks.
import sys
from pathlib import Path

ROOT      = Path(__file__).resolve().parent
SRC       = ROOT / "src"
INGESTOR  = SRC / "ingestor"
PARSERS   = INGESTOR / "parsers"
ML_MODELS = SRC / "ml_models"
GNN       = SRC / "gnn_fusion"
SIMULATOR = ROOT / "simulator"

for p in [SRC, INGESTOR, PARSERS, ML_MODELS, GNN, SIMULATOR]:
    sys.path.insert(0, str(p))