"""
config.py — Where the API finds the trained project and the raw NASA files.

Defaults assume this layout (the same as Google Drive):
    <somewhere>/Predictive_Maintenance_RUL/   <- this project (api/, src/, models/, reports/)
    <somewhere>/CMAPSSData/                   <- raw NASA text files (train_FD001.txt, ...)
Override either one with environment variables RUL_PROJECT_DIR and CMAPSS_RAW_DIR.
"""
import os
import sys
from pathlib import Path

PROJECT_DIR = Path(os.environ.get("RUL_PROJECT_DIR", Path(__file__).resolve().parents[1]))
_BUNDLED_RAW = PROJECT_DIR / "data" / "CMAPSSData"          # NASA test + RUL files shipped with the repo
RAW_DIR = Path(os.environ.get("CMAPSS_RAW_DIR",
                              _BUNDLED_RAW if _BUNDLED_RAW.exists() else PROJECT_DIR.parent / "CMAPSSData"))
MODELS_DIR = PROJECT_DIR / "models"
REPORTS_DIR = PROJECT_DIR / "reports"
DATASETS = ["FD001", "FD002", "FD003", "FD004"]

# Interval coverage used for the "uncertainty range" (split conformal on validation residuals)
INTERVAL_COVERAGE = 0.90

# Demonstration risk thresholds on predicted RUL (cycles). NOT airline maintenance rules.
RISK_LEVELS = [(20, "Critical"), (50, "High"), (90, "Moderate")]   # below 20 -> Critical, ... ; otherwise Low

if str(PROJECT_DIR) not in sys.path:          # so `from src...` works
    sys.path.insert(0, str(PROJECT_DIR))
