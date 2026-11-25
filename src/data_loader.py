"""
data_loader.py — Loading NASA C-MAPSS files (read-only).

Why this file exists: every notebook (EDA, preprocessing, training) and later the
FastAPI backend must read the data in *exactly* the same way, with the same
column names and order. Putting it in one module avoids copy-paste drift.
"""
from pathlib import Path
import numpy as np
import pandas as pd

DATASETS = ["FD001", "FD002", "FD003", "FD004"]

ID_COLS      = ["engine_id", "cycle"]
SETTING_COLS = [f"op_setting_{i}" for i in range(1, 4)]
SENSOR_COLS  = [f"sensor_{i}" for i in range(1, 22)]
COLUMNS      = ID_COLS + SETTING_COLS + SENSOR_COLS          # 26 columns
FEATURE_CANDIDATES = SETTING_COLS + SENSOR_COLS              # 24 candidate inputs


def load_cmapss_file(path) -> pd.DataFrame:
    """Load a train_/test_ file. Whitespace separator handles trailing spaces."""
    path = Path(path)
    df = pd.read_csv(path, sep=r"\s+", header=None)
    if df.shape[1] != len(COLUMNS):
        raise ValueError(f"{path.name}: expected {len(COLUMNS)} columns, got {df.shape[1]}")
    df.columns = COLUMNS
    df["engine_id"] = df["engine_id"].astype(int)
    df["cycle"] = df["cycle"].astype(int)
    return df


def load_rul_file(path) -> pd.DataFrame:
    """Load an RUL_ file: line i = true RUL of test engine i at its last observed cycle."""
    path = Path(path)
    df = pd.read_csv(path, sep=r"\s+", header=None)
    if df.shape[1] != 1:
        raise ValueError(f"{path.name}: expected 1 column, got {df.shape[1]}")
    df.columns = ["true_rul"]
    df.insert(0, "engine_id", np.arange(1, len(df) + 1))
    return df


def load_dataset(fd: str, raw_dir) -> dict:
    """Return {'train', 'test', 'rul'} DataFrames for one sub-dataset."""
    if fd not in DATASETS:
        raise ValueError(f"Unknown dataset {fd!r}; expected one of {DATASETS}")
    raw_dir = Path(raw_dir)
    return {
        "train": load_cmapss_file(raw_dir / f"train_{fd}.txt"),
        "test":  load_cmapss_file(raw_dir / f"test_{fd}.txt"),
        "rul":   load_rul_file(raw_dir / f"RUL_{fd}.txt"),
    }
