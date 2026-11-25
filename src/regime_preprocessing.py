"""
regime_preprocessing.py — Operating-regime-aware preprocessing for multi-condition C-MAPSS (FD002, FD004).

Why a separate file: FD001/FD003 run in ONE operating condition and use CMAPSSPreprocessor
(src/preprocessing.py) unchanged. FD002/FD004 switch between 6 conditions almost every cycle, and the
condition shifts every sensor far more than wear does. RegimePreprocessor removes that shift by
standardising each sensor separately inside each regime. Both classes share the same interface
(fit / transform / save / load / feature_cols), so the rest of the project treats them the same.
Use load_preprocessor(directory) to load whichever kind was saved.

Everything learned (regime centres, per-regime means/stds) is fitted on TRAINING engines only.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.data_loader import ID_COLS, SENSOR_COLS, SETTING_COLS
from src.preprocessing import TARGET, CMAPSSPreprocessor

REGIME_ROUNDING = {"op_setting_1": 0, "op_setting_2": 2, "op_setting_3": 0}


class RegimePreprocessor:
    """Fit on training engines -> per-regime z-scores for train / validation / test / API input."""

    kind = "regime"

    def __init__(self, feature_cols=None, min_unique_within_regime: int = 3, tolerance_factor: float = 3.0):
        self.feature_cols = list(feature_cols) if feature_cols is not None else None
        self.min_unique_within_regime = min_unique_within_regime
        self.tolerance_factor = tolerance_factor
        self.dropped_features = {}
        self.regime_centers = None      # DataFrame index=regime (1..K), columns=SETTING_COLS
        self.regime_tolerance = None    # max allowed |setting - centre| per setting
        self.scalers = {}               # regime -> fitted StandardScaler

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _validate(df, cols):
        missing = [c for c in ID_COLS + list(cols) if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        bad = [c for c in cols if not pd.api.types.is_numeric_dtype(df[c])]
        if bad:
            raise ValueError(f"Non-numeric columns: {bad}")
        n_nan = int(df[list(cols)].isna().sum().sum())
        if n_nan:
            raise ValueError(f"Input contains {n_nan} missing values; C-MAPSS pipeline does not impute.")

    def assign_regime(self, df: pd.DataFrame) -> pd.Series:
        """Nearest learned regime centre for every row. Raises if a row is far from every centre
        (an operating condition never seen in training -> the model must not guess)."""
        if self.regime_centers is None:
            raise RuntimeError("Call fit() first.")
        self._validate(df, SETTING_COLS)
        x = df[SETTING_COLS].to_numpy(float)
        c = self.regime_centers.to_numpy(float)
        span = c.max(axis=0) - c.min(axis=0)
        span[span == 0] = 1.0
        dist = (((x[:, None, :] - c[None, :, :]) / span) ** 2).sum(axis=2)
        idx = dist.argmin(axis=1)
        dev = np.abs(x - c[idx])
        bad = (dev > self.regime_tolerance.to_numpy(float)).any(axis=1)
        if bad.any():
            raise ValueError(f"{int(bad.sum())} row(s) have operational settings outside every known "
                             f"operating regime (first: {x[bad][0].round(4).tolist()}).")
        return pd.Series(self.regime_centers.index.to_numpy()[idx], index=df.index, name="regime")

    # ------------------------------------------------------------------ fit / transform
    def _fit_regimes(self, train_df):
        rounded = train_df[SETTING_COLS].round(REGIME_ROUNDING)
        keys = rounded.apply(tuple, axis=1)
        order = sorted(keys.unique())
        label = keys.map({k: i + 1 for i, k in enumerate(order)})
        self.regime_centers = train_df[SETTING_COLS].groupby(label).mean()
        self.regime_centers.index.name = "regime"
        max_dev = (train_df[SETTING_COLS] - self.regime_centers.loc[label].to_numpy()).abs().max()
        self.regime_tolerance = (max_dev * self.tolerance_factor).clip(lower=1e-3)
        return label

    def fit(self, train_df: pd.DataFrame):
        self._validate(train_df, SETTING_COLS)
        regime = self._fit_regimes(train_df)
        n_within = train_df[SENSOR_COLS].groupby(regime).nunique().max()
        if self.feature_cols is None:      # default rule: drop sensors with < min_unique values in every regime
            self.feature_cols = [c for c in SENSOR_COLS if n_within[c] >= self.min_unique_within_regime]
        self.dropped_features = {}
        for c in SENSOR_COLS:
            if c in self.feature_cols:
                continue
            if n_within[c] == 1:
                self.dropped_features[c] = "constant within every operating regime"
            elif n_within[c] < self.min_unique_within_regime:
                self.dropped_features[c] = f"at most {n_within[c]} distinct values within any regime"
            else:
                self.dropped_features[c] = "excluded by validation feature-set experiment"
        self._validate(train_df, self.feature_cols)
        for col in SETTING_COLS:
            self.dropped_features.setdefault(col, "used only to identify the operating regime")
        self.scalers = {int(r): StandardScaler().fit(g[self.feature_cols])
                        for r, g in train_df.groupby(regime)}
        return self

    def transform(self, df: pd.DataFrame, keep_regime: bool = False) -> pd.DataFrame:
        if not self.scalers:
            raise RuntimeError("Call fit() on training data before transform().")
        self._validate(df, self.feature_cols)
        regime = self.assign_regime(df)
        keep = ID_COLS + ([TARGET] if TARGET in df.columns else [])
        out = df[keep].copy()
        scaled = np.empty((len(df), len(self.feature_cols)))
        for r, scaler in self.scalers.items():
            m = (regime == r).to_numpy()
            if m.any():
                scaled[m] = scaler.transform(df.loc[m, self.feature_cols])
        out[self.feature_cols] = scaled
        if keep_regime:
            out["regime"] = regime.to_numpy()
        return out

    # ------------------------------------------------------------------ persistence
    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.scalers, directory / "regime_scalers.joblib")
        config = {"kind": self.kind, "scaler": "standard (per regime)", "feature_cols": self.feature_cols,
                  "dropped_features": self.dropped_features, "regime_rounding": REGIME_ROUNDING,
                  "regime_centers": {int(r): row.round(6).to_dict() for r, row in self.regime_centers.iterrows()},
                  "regime_tolerance": self.regime_tolerance.round(6).to_dict(),
                  "min_unique_within_regime": self.min_unique_within_regime,
                  "tolerance_factor": self.tolerance_factor}
        (directory / "preprocessing_config.json").write_text(json.dumps(config, indent=2))

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        config = json.loads((directory / "preprocessing_config.json").read_text())
        obj = cls(feature_cols=config["feature_cols"],
                  min_unique_within_regime=config["min_unique_within_regime"],
                  tolerance_factor=config["tolerance_factor"])
        obj.dropped_features = config["dropped_features"]
        centers = pd.DataFrame.from_dict(config["regime_centers"], orient="index")
        centers.index = centers.index.astype(int)
        obj.regime_centers = centers[SETTING_COLS].sort_index().rename_axis("regime")
        obj.regime_tolerance = pd.Series(config["regime_tolerance"])[SETTING_COLS]
        obj.scalers = {int(k): v for k, v in joblib.load(directory / "regime_scalers.joblib").items()}
        return obj


def load_preprocessor(directory):
    """Load whichever preprocessor was saved in `directory` (regime-aware or single-condition)."""
    config = json.loads((Path(directory) / "preprocessing_config.json").read_text())
    if config.get("kind") == "regime":
        return RegimePreprocessor.load(directory)
    return CMAPSSPreprocessor.load(directory)
