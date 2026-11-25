"""
evaluation.py — Metrics for RUL prediction (shared by every model and dataset).

Sign convention (used everywhere in this project):
    error d = predicted_RUL - true_RUL
    d > 0  -> LATE prediction  (model thinks engine has MORE life than it does)  -> dangerous
    d < 0  -> EARLY prediction (model thinks engine has LESS life than it does) -> wasteful
"""
import numpy as np


def rmse(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def mae(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.mean(np.abs(y_pred - y_true)))


def nasa_score(y_true, y_pred) -> float:
    """PHM08 asymmetric score (Saxena et al., 2008). Lower is better.
    s = sum( exp(-d/13) - 1 ) for d < 0 (early),  sum( exp(d/10) - 1 ) for d >= 0 (late).
    Late predictions are penalised more heavily than early ones."""
    d = np.asarray(y_pred, float) - np.asarray(y_true, float)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)))


def all_metrics(y_true, y_pred) -> dict:
    return {"RMSE": rmse(y_true, y_pred), "MAE": mae(y_true, y_pred),
            "NASA_score": nasa_score(y_true, y_pred)}
