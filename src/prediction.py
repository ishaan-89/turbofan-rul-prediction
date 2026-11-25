"""
prediction.py — Load a saved model bundle and predict RUL from RAW engine history.

Why this file exists: evaluation (Phase 6), the FastAPI backend and the tests must all
turn raw sensor rows into a prediction in exactly the same way:
    raw rows -> saved preprocessing (validate + scale) -> saved feature recipe -> saved model
Everything is read from disk (models/<FD>/...), nothing is re-fitted.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src import feature_engineering as fe
from src.preprocessing import CMAPSSPreprocessor

try:                                    # regime-aware datasets (FD002 / FD004)
    from src.regime_preprocessing import load_preprocessor
except ImportError:                     # project without FD002/FD004 yet
    load_preprocessor = CMAPSSPreprocessor.load

MODEL_FOLDERS = {"Linear Regression": "linear_regression", "Random Forest": "random_forest",
                 "XGBoost": "xgboost", "LSTM": "lstm"}


class RULPredictor:
    """One trained model + its preprocessing + its feature recipe, loaded from models/<FD>/."""

    def __init__(self, dataset_dir, model_name: str, device: str = "cpu"):
        dataset_dir = Path(dataset_dir)
        self.name = model_name
        self.dir = dataset_dir / MODEL_FOLDERS[model_name]
        self.meta = json.loads((self.dir / "metadata.json").read_text())
        self.prep = load_preprocessor(dataset_dir / "preprocessing")     # single-regime or per-regime
        self.feat_cfg = json.loads((dataset_dir / "feature_config.json").read_text())
        self.cap = float(self.meta["rul_cap"])
        self.device = device
        self.base = self.feat_cfg["base_features"]
        if self.meta["input"] == "sequence":
            from src.model_training import load_lstm
            self.model, _ = load_lstm(self.dir, device)
            self.seq_len = int(self.meta["seq_len"])
            self.min_history = self.seq_len
            # LSTM inputs: plain sensors (FD001) or sensors + causal rolling means (FD002/FD004)
            self.seq_features = self.meta.get("features", self.base)
            self.seq_rolling_window = self.feat_cfg["lstm"].get("rolling_window")
        else:
            if self.meta["model_file"].endswith(".json"):          # XGBoost
                from xgboost import XGBRegressor
                self.model = XGBRegressor()
                self.model.load_model(self.dir / self.meta["model_file"])
                self.model.set_params(device=device)
                self.model.get_booster().set_param({"device": device})
            else:                                                   # scikit-learn
                self.model = joblib.load(self.dir / self.meta["model_file"])
            tab = self.feat_cfg["tabular"]
            self.window, self.include_cycle = tab["window"], tab["include_cycle"]
            self.feature_order = tab["feature_order"]
            self.min_history = 1

    # -------------------------------------------------------------- core
    def model_inputs(self, raw_df: pd.DataFrame, points: pd.DataFrame):
        """Raw history -> model-ready inputs for each (engine_id, cut_cycle) in `points`."""
        scaled = self.prep.transform(raw_df)                       # validates columns / NaN, then scales
        if self.meta["input"] == "sequence":
            if self.seq_rolling_window:
                scaled = fe.add_rolling_features(scaled, self.base, self.seq_rolling_window)
            return fe.sequences_at(scaled, self.seq_features, self.seq_len, points)
        feats, _ = fe.build_tabular(scaled, self.base, self.window, self.include_cycle)
        return fe.rows_at(feats, points)[self.feature_order]

    def predict_inputs(self, X) -> np.ndarray:
        """Model-ready inputs -> RUL predictions (clipped at 0)."""
        if self.meta["input"] == "sequence":
            from src.model_training import lstm_predict
            return lstm_predict(self.model, X, self.cap, self.device)
        return np.clip(self.model.predict(X), 0, None)

    def predict_at(self, raw_df: pd.DataFrame, points: pd.DataFrame) -> np.ndarray:
        return self.predict_inputs(self.model_inputs(raw_df, points))

    def predict_last(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """Predict at each engine's LAST observed cycle (what the official test set asks for)."""
        last = raw_df.groupby("engine_id")["cycle"].max().rename("cut_cycle").reset_index()
        short = last[last["cut_cycle"] < self.min_history]
        if len(short):
            raise ValueError(f"{self.name} needs >= {self.min_history} cycles; engines too short: "
                             f"{short.engine_id.tolist()}")
        last["predicted_rul"] = self.predict_at(raw_df, last)
        return last
