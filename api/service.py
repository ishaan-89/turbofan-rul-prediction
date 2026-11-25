"""
service.py — Everything the API does, without any web code (so tests can call it directly).

    raw engine rows -> saved preprocessing -> saved features -> saved model -> RUL
                    -> uncertainty range, Health Index, risk level, SHAP explanation

Nothing is re-trained here. Every model, scaler and feature list is read from models/<FD>/.
"""
import json
import math
from functools import lru_cache

import numpy as np
import pandas as pd

from api import config
from src.data_loader import COLUMNS, load_cmapss_file, load_rul_file
from src.prediction import MODEL_FOLDERS, RULPredictor

MODEL_NAMES = list(MODEL_FOLDERS)


class InputError(ValueError):
    """Bad user input (missing columns, NaN, too-short history, unknown regime...). -> HTTP 422"""


# --------------------------------------------------------------------------- small helpers
def check_dataset(fd: str) -> str:
    fd = fd.upper()
    if fd not in config.DATASETS:
        raise KeyError(f"Unknown dataset '{fd}'. Choose one of {config.DATASETS}.")
    return fd


def selected_model(fd: str) -> dict:
    return json.loads((config.MODELS_DIR / fd / "selected_model.json").read_text())


def health_index(rul: float, cap: float) -> int:
    """Custom project metric: predicted RUL as a % of the cap (100 = at or above the cap, 0 = failure)."""
    return int(round(100 * min(max(rul, 0.0), cap) / cap))


def risk_level(rul: float) -> str:
    for limit, name in config.RISK_LEVELS:
        if rul < limit:
            return name
    return "Low"


def records_to_frame(records) -> pd.DataFrame:
    """List of row dicts (or a DataFrame) in C-MAPSS column names -> validated DataFrame."""
    df = pd.DataFrame(records) if not isinstance(records, pd.DataFrame) else records.copy()
    if df.empty:
        raise InputError("No rows received.")
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise InputError(f"Missing required columns: {missing}")
    df = df[COLUMNS].copy()
    for c in COLUMNS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    n_nan = int(df.isna().sum().sum())
    if n_nan:
        raise InputError(f"Input has {n_nan} missing or non-numeric values. The pipeline does not guess values.")
    df["engine_id"] = df["engine_id"].astype(int)
    df["cycle"] = df["cycle"].astype(int)
    if df.duplicated(["engine_id", "cycle"]).any():
        raise InputError("Duplicate (engine_id, cycle) rows found.")
    return df.sort_values(["engine_id", "cycle"]).reset_index(drop=True)


def parse_upload(content: bytes) -> pd.DataFrame:
    """Accept either a raw NASA .txt file (26 space-separated columns) or a CSV with C-MAPSS headers."""
    import io
    text = content.decode("utf-8", errors="replace")
    first = text.strip().splitlines()[0] if text.strip() else ""
    if "engine_id" in first:
        return records_to_frame(pd.read_csv(io.StringIO(text)))
    df = pd.read_csv(io.StringIO(text), sep=r"\s+", header=None)
    if df.shape[1] != len(COLUMNS):
        raise InputError(f"Expected {len(COLUMNS)} columns (NASA format) or a CSV with headers; got {df.shape[1]} columns.")
    df.columns = COLUMNS
    return records_to_frame(df)


# --------------------------------------------------------------------------- cached loaders
@lru_cache(maxsize=None)
def predictor(fd: str, model: str) -> RULPredictor:
    if model not in MODEL_FOLDERS:
        raise KeyError(f"Unknown model '{model}'. Choose one of {MODEL_NAMES}.")
    return RULPredictor(config.MODELS_DIR / fd, model)


@lru_cache(maxsize=None)
def conformal_halfwidth(fd: str, model: str) -> float:
    """Split-conformal half-width from VALIDATION residuals (target capped like training).
    Note: validation engines were also used to pick the model, so coverage may be slightly optimistic;
    the dashboard reports the coverage actually reached on the official test set."""
    v = pd.read_csv(config.REPORTS_DIR / "model_results" / fd / "validation_predictions.csv")
    cap = predictor(fd, model).cap
    r = np.sort(np.abs(np.minimum(v["true_rul"], cap) - v[model]).to_numpy())
    k = min(len(r) - 1, math.ceil((len(r) + 1) * config.INTERVAL_COVERAGE) - 1)
    return float(r[k])


@lru_cache(maxsize=None)
def raw_test(fd: str):
    """Official NASA test engines + their true RUL (true RUL is shown only for demonstration)."""
    path = config.RAW_DIR / f"test_{fd}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Raw file not found: {path}. Set CMAPSS_RAW_DIR to the CMAPSSData folder.")
    return load_cmapss_file(path), load_rul_file(config.RAW_DIR / f"RUL_{fd}.txt")


# --------------------------------------------------------------------------- main operations
def predict(fd: str, df: pd.DataFrame, model: str | None = None) -> dict:
    fd = check_dataset(fd)
    model = model or selected_model(fd)["selected_model"]
    p = predictor(fd, model)
    try:
        out = p.predict_last(df)
    except ValueError as e:                       # too-short history, unknown regime, bad columns
        raise InputError(str(e)) from e
    q = conformal_halfwidth(fd, model)
    n_cycles = df.groupby("engine_id")["cycle"].size()
    engines = []
    for r in out.itertuples():
        rul = float(r.predicted_rul)
        engines.append({
            "engine_id": int(r.engine_id), "last_cycle": int(r.cut_cycle), "n_cycles": int(n_cycles[r.engine_id]),
            "predicted_rul": round(rul, 1),
            "rul_lower": round(max(0.0, rul - q), 1), "rul_upper": round(min(p.cap, rul + q), 1),
            "at_cap": bool(rul >= p.cap - 0.5),
            "health_index": health_index(rul, p.cap), "risk_level": risk_level(rul)})
    return {"dataset": fd, "model": model, "rul_cap": p.cap, "interval_coverage": config.INTERVAL_COVERAGE,
            "interval_halfwidth": round(q, 2), "engines": engines}


def fleet(fd: str) -> dict:
    """Predict every official test engine and attach its true RUL (for the demo fleet view)."""
    fd = check_dataset(fd)
    test, rul = raw_test(fd)
    res = predict(fd, test)
    truth = dict(zip(rul.engine_id, rul.true_rul))
    for e in res["engines"]:
        e["true_rul"] = int(truth[e["engine_id"]])
    cap = res["rul_cap"]
    inside = [e["rul_lower"] <= min(e["true_rul"], cap) <= e["rul_upper"] for e in res["engines"]]
    res["test_interval_coverage"] = round(float(np.mean(inside)), 3)
    return res


def explain(fd: str, df: pd.DataFrame, engine_id: int, top_k: int = 12) -> dict:
    """SHAP values for one engine's last cycle (tree models only)."""
    import shap
    fd = check_dataset(fd)
    model = selected_model(fd)["selected_model"]
    p = predictor(fd, model)
    if p.meta["input"] != "tabular" or model == "Linear Regression":
        raise InputError(f"SHAP explanations are available for tree models; {fd} uses {model}.")
    one = df[df.engine_id == engine_id]
    if one.empty:
        raise InputError(f"engine_id {engine_id} not in input.")
    point = pd.DataFrame({"engine_id": [engine_id], "cut_cycle": [int(one.cycle.max())]})
    try:
        X = p.model_inputs(one, point)
    except ValueError as e:
        raise InputError(str(e)) from e
    expl = _explainer(fd, model)
    sv = np.asarray(expl.shap_values(X))[0]
    base = float(np.ravel(expl.expected_value)[0])
    feats = pd.DataFrame({"feature": p.feature_order, "value": X.iloc[0].to_numpy(), "shap": sv})
    feats["sensor"] = feats.feature.str.extract(r"^(sensor_\d+|cycle)")[0].fillna(feats.feature)
    by_sensor = feats.groupby("sensor").shap.sum().sort_values(key=np.abs, ascending=False)
    top = feats.reindex(feats.shap.abs().sort_values(ascending=False).index).head(top_k)
    raw_pred = base + float(sv.sum())
    return {"dataset": fd, "model": model, "engine_id": int(engine_id), "base_value": round(base, 2),
            "prediction": round(float(np.clip(raw_pred, 0, None)), 1),
            "top_features": [{"feature": r.feature, "scaled_value": round(float(r.value), 3), "shap": round(float(r.shap), 3)}
                             for r in top.itertuples()],
            "by_sensor": [{"sensor": k, "shap": round(float(v), 3)} for k, v in by_sensor.items()]}


@lru_cache(maxsize=None)
def _explainer(fd: str, model: str):
    import shap
    return shap.TreeExplainer(predictor(fd, model).model)


def dataset_info(fd: str) -> dict:
    fd = check_dataset(fd)
    sel = selected_model(fd)
    p = predictor(fd, sel["selected_model"])
    feat = json.loads((config.MODELS_DIR / fd / "feature_config.json").read_text())
    test = json.loads((config.REPORTS_DIR / "model_results" / fd / "test_evaluation.json").read_text()) \
        if (config.REPORTS_DIR / "model_results" / fd / "test_evaluation.json").exists() else None
    return {"dataset": fd, "selected_model": sel["selected_model"], "selection_rule": sel["rule"],
            "rul_cap": p.cap, "window": feat["tabular"]["window"], "base_features": feat["base_features"],
            "n_model_features": len(feat["tabular"]["feature_order"]),
            "validation_rmse": round(float(next(v for k, v in sel["validation_metrics"].items() if k.startswith("RMSE (≤"))), 2),
            "test_evaluation": test}


def comparison() -> list[dict]:
    df = pd.read_csv(config.REPORTS_DIR / "final_comparison" / "all_models_all_datasets.csv")
    return json.loads(df.to_json(orient="records"))
