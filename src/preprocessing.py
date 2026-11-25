"""
preprocessing.py — Leakage-safe preprocessing for NASA C-MAPSS.

Why this file exists: training notebooks, evaluation and the FastAPI backend must
transform data in *exactly* the same way. Every "learned" step (feature selection,
scaling) is fitted on TRAINING engines only, then frozen and re-applied elsewhere.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from src.data_loader import FEATURE_CANDIDATES, ID_COLS, SETTING_COLS

TARGET = "RUL"


# ----------------------------------------------------------------------------- labels
def add_train_rul(df: pd.DataFrame, cap=None) -> pd.DataFrame:
    """RUL = (engine's last cycle) - (current cycle). Optional upper cap (piecewise-linear target).
    Valid ONLY for run-to-failure (training) trajectories."""
    out = df.copy()
    last_cycle = out.groupby("engine_id")["cycle"].transform("max")
    out[TARGET] = (last_cycle - out["cycle"]).astype(float)
    if cap is not None:
        out[TARGET] = out[TARGET].clip(upper=cap)
    return out


def add_test_rul(test_df: pd.DataFrame, rul_df: pd.DataFrame) -> pd.DataFrame:
    """FINAL EVALUATION ONLY. RUL at every test row = true RUL at last cycle + cycles remaining
    until that last cycle. Never use for training, tuning or model selection."""
    out = test_df.copy()
    last_cycle = out.groupby("engine_id")["cycle"].transform("max")
    rul_at_last = out["engine_id"].map(rul_df.set_index("engine_id")["true_rul"])
    if rul_at_last.isna().any():
        raise ValueError("Some test engines have no entry in the RUL file.")
    out[TARGET] = (rul_at_last + (last_cycle - out["cycle"])).astype(float)
    return out


# ----------------------------------------------------------------------------- splitting
def split_engines(df: pd.DataFrame, val_fraction: float = 0.2, seed: int = 42):
    """Split by ENGINE, not by row: every row of an engine lands in exactly one split."""
    ids = np.sort(df["engine_id"].unique())
    n_val = int(round(len(ids) * val_fraction))
    rng = np.random.default_rng(seed)
    val_ids = np.sort(rng.choice(ids, size=n_val, replace=False))
    train_ids = np.setdiff1d(ids, val_ids)
    return train_ids.tolist(), val_ids.tolist()


# ----------------------------------------------------------------------------- feature selection
def count_regimes(df: pd.DataFrame) -> int:
    """Number of distinct operating conditions, found by rounding settings to a coarse grid."""
    rounded = df[SETTING_COLS].round({"op_setting_1": 0, "op_setting_2": 2, "op_setting_3": 0})
    return int(len(rounded.drop_duplicates()))


def select_features(train_df: pd.DataFrame, candidates=FEATURE_CANDIDATES, min_unique: int = 3):
    """Data-driven feature selection, computed on the TRAINING split only.
    Rules (documented in Phase 3):
      1. Drop features with fewer than `min_unique` distinct values (constant / near-constant).
      2. If the data has a single operating regime, drop the operational settings
         (they are then only measurement noise around one set-point).
    Returns (kept_features_in_original_order, {dropped_feature: reason})."""
    dropped = {}
    n_unique = train_df[candidates].nunique()
    for col in candidates:
        if n_unique[col] < min_unique:
            dropped[col] = f"only {n_unique[col]} distinct value(s) in training split"
    if count_regimes(train_df) == 1:
        for col in SETTING_COLS:
            dropped.setdefault(col, "single operating regime -> setting is noise")
    kept = [c for c in candidates if c not in dropped]
    return kept, dropped


# ----------------------------------------------------------------------------- preprocessor
_SCALERS = {"standard": StandardScaler, "minmax": MinMaxScaler}


class CMAPSSPreprocessor:
    """Fit on training engines -> transform train / validation / test / API input identically."""

    def __init__(self, scaler: str = "standard", feature_cols=None):
        if scaler not in _SCALERS:
            raise ValueError(f"scaler must be one of {list(_SCALERS)}")
        self.scaler_name = scaler
        self.feature_cols = list(feature_cols) if feature_cols is not None else None
        self.dropped_features = {}
        self.scaler = None

    # -- validation shared by fit and transform
    def _validate(self, df: pd.DataFrame, cols):
        missing = [c for c in ID_COLS + list(cols) if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        bad_types = [c for c in cols if not pd.api.types.is_numeric_dtype(df[c])]
        if bad_types:
            raise ValueError(f"Non-numeric columns: {bad_types}")
        n_nan = int(df[list(cols)].isna().sum().sum())
        if n_nan:
            raise ValueError(f"Input contains {n_nan} missing values; C-MAPSS pipeline does not impute.")

    def fit(self, train_df: pd.DataFrame):
        if self.feature_cols is None:
            self.feature_cols, self.dropped_features = select_features(train_df)
        self._validate(train_df, self.feature_cols)
        self.scaler = _SCALERS[self.scaler_name]().fit(train_df[self.feature_cols])
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.scaler is None:
            raise RuntimeError("Call fit() on training data before transform().")
        self._validate(df, self.feature_cols)
        keep = ID_COLS + ([TARGET] if TARGET in df.columns else [])
        out = df[keep].copy()
        scaled = self.scaler.transform(df[self.feature_cols])   # DataFrame in -> column order enforced
        out[self.feature_cols] = scaled
        return out

    # -- persistence: scaler (joblib) + human-readable config (JSON)
    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.scaler, directory / "scaler.joblib")
        config = {"scaler": self.scaler_name, "feature_cols": self.feature_cols,
                  "dropped_features": self.dropped_features}
        (directory / "preprocessing_config.json").write_text(json.dumps(config, indent=2))

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        config = json.loads((directory / "preprocessing_config.json").read_text())
        obj = cls(scaler=config["scaler"], feature_cols=config["feature_cols"])
        obj.dropped_features = config["dropped_features"]
        obj.scaler = joblib.load(directory / "scaler.joblib")
        return obj


# ----------------------------------------------------------------------------- test-like validation
def make_eval_cut_points(df: pd.DataFrame, n_cuts: int = 10, min_history: int = 30, seed: int = 42):
    """Test-like validation view. For each engine, pick `n_cuts` random 'stop' cycles in
    [min_history, last_cycle]; we pretend the engine's data ends there (as in the official test set)
    and score the prediction made at that cycle against the TRUE, UNCAPPED RUL.
    Returns DataFrame[engine_id, cut_cycle, true_rul]."""
    rng = np.random.default_rng(seed)
    rows = []
    for eid, g in df.groupby("engine_id"):
        last = int(g["cycle"].max())
        lo = min(min_history, last)
        cuts = rng.choice(np.arange(lo, last + 1), size=min(n_cuts, last - lo + 1), replace=False)
        rows += [(int(eid), int(c), float(last - c)) for c in np.sort(cuts)]
    return pd.DataFrame(rows, columns=["engine_id", "cut_cycle", "true_rul"])
