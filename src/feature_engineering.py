"""
feature_engineering.py — Leakage-free (causal) features for C-MAPSS.

Why this file exists: a single sensor reading is noisy; the *recent history* of an engine
carries the degradation signal. Every feature here at cycle t uses ONLY cycles <= t of the
SAME engine, so it can be computed identically in training, validation, test and the API.
Nothing in this file is fitted, so it cannot leak validation/test information.
"""
import numpy as np
import pandas as pd


# ----------------------------------------------------------------------------- tabular features
def add_rolling_features(df: pd.DataFrame, base_cols, window: int) -> pd.DataFrame:
    """For each base column add, per engine, over the last `window` cycles (current included):
         <col>_mean_w   rolling mean          (smooths noise)
         <col>_std_w    rolling std           (local volatility)
         <col>_slope_w  rolling linear slope  (direction + speed of degradation, units per cycle)
    At an engine's first cycles the window is shorter (min_periods=1); std/slope that are
    undefined there (only one point) are set to 0 = 'no variation observed yet'."""
    if not df.index.is_unique:
        raise ValueError("DataFrame index must be unique")
    df = df.sort_values(["engine_id", "cycle"])
    eng = df["engine_id"]

    def rmean(s):   # rolling mean of a series, restarted for every engine
        return s.groupby(eng).transform(lambda x: x.rolling(window, min_periods=1).mean())

    t = df["cycle"].astype(float)
    t_mean = rmean(t)
    var_t = rmean(t * t) - t_mean ** 2           # variance of cycle numbers inside the window

    new_cols = {}
    for c in base_cols:
        x = df[c].astype(float)
        x_mean = rmean(x)
        x_std = x.groupby(eng).transform(lambda s: s.rolling(window, min_periods=1).std())
        slope = (rmean(x * t) - x_mean * t_mean) / var_t.where(var_t > 1e-12)   # least-squares slope
        new_cols[f"{c}_mean_{window}"] = x_mean
        new_cols[f"{c}_std_{window}"] = x_std.fillna(0.0)
        new_cols[f"{c}_slope_{window}"] = slope.fillna(0.0)
    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


def tabular_feature_names(base_cols, window: int, include_cycle: bool):
    """Exact, fixed feature ORDER for the traditional models (LR / RF / XGBoost)."""
    names = list(base_cols)
    if window:
        for c in base_cols:
            names += [f"{c}_mean_{window}", f"{c}_std_{window}", f"{c}_slope_{window}"]
    if include_cycle:
        names.append("cycle")
    return names


def build_tabular(df_scaled: pd.DataFrame, base_cols, window: int, include_cycle: bool):
    """Return (feature_frame_with_ids, ordered_feature_names)."""
    out = add_rolling_features(df_scaled, base_cols, window) if window else df_scaled.copy()
    return out, tabular_feature_names(base_cols, window, include_cycle)


def rows_at(feature_df: pd.DataFrame, points: pd.DataFrame) -> pd.DataFrame:
    """Pick the feature row at each (engine_id, cut_cycle) in `points`, keeping the order of `points`."""
    out = points[["engine_id", "cut_cycle"]].merge(
        feature_df, left_on=["engine_id", "cut_cycle"], right_on=["engine_id", "cycle"], how="left")
    if out["cycle"].isna().any():
        raise ValueError("Some requested (engine_id, cycle) points do not exist in the data.")
    return out


# ----------------------------------------------------------------------------- LSTM sequences
def make_sequences(df_scaled: pd.DataFrame, feature_cols, seq_len: int, target: str = "RUL"):
    """All sliding windows of `seq_len` consecutive cycles inside each engine.
    Window ending at cycle t uses cycles t-seq_len+1 .. t; label = target at cycle t.
    Engines/cycles with fewer than seq_len cycles of history produce no window.
    Returns X [n, seq_len, n_features] float32, y [n], meta DataFrame[engine_id, end_cycle]."""
    X, y, meta = [], [], []
    for eid, g in df_scaled.sort_values(["engine_id", "cycle"]).groupby("engine_id"):
        arr = g[feature_cols].to_numpy(np.float32)
        cyc = g["cycle"].to_numpy()
        tgt = g[target].to_numpy(np.float32) if target in g else np.full(len(g), np.nan, np.float32)
        for end in range(seq_len - 1, len(g)):
            X.append(arr[end - seq_len + 1:end + 1])
            y.append(tgt[end])
            meta.append((int(eid), int(cyc[end])))
    X = np.stack(X) if X else np.empty((0, seq_len, len(feature_cols)), np.float32)
    return X, np.asarray(y, np.float32), pd.DataFrame(meta, columns=["engine_id", "end_cycle"])


def sequences_at(df_scaled: pd.DataFrame, feature_cols, seq_len: int, points: pd.DataFrame):
    """One window per (engine_id, cut_cycle) row of `points`, ending exactly at that cycle.
    Used for test-like validation, official test (last cycle) and API requests."""
    by_engine = {eid: g.sort_values("cycle") for eid, g in df_scaled.groupby("engine_id")}
    X = []
    for eid, cut in zip(points["engine_id"], points["cut_cycle"]):
        g = by_engine[eid]
        hist = g[g["cycle"] <= cut][feature_cols].to_numpy(np.float32)
        if len(hist) < seq_len:
            raise ValueError(f"Engine {eid} has only {len(hist)} cycles up to {cut}; need {seq_len}.")
        X.append(hist[-seq_len:])
    return np.stack(X)
