"""
main.py — FastAPI backend for RUL prediction.

Run from the project folder:
    uvicorn api.main:app --port 8000
Interactive docs: http://127.0.0.1:8000/docs
"""
import time

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from api import config, service

app = FastAPI(
    title="Turbofan RUL Prediction API",
    description="Predicts Remaining Useful Life (cycles) for NASA C-MAPSS engines with the models selected "
                "on validation data. Research prototype on simulated data, not certified aviation software.",
    version="1.0.0",
)


# --------------------------------------------------------------------------- request / response shapes
class EngineRow(BaseModel):
    engine_id: int
    cycle: int
    op_setting_1: float
    op_setting_2: float
    op_setting_3: float
    sensor_1: float; sensor_2: float; sensor_3: float; sensor_4: float; sensor_5: float
    sensor_6: float; sensor_7: float; sensor_8: float; sensor_9: float; sensor_10: float
    sensor_11: float; sensor_12: float; sensor_13: float; sensor_14: float; sensor_15: float
    sensor_16: float; sensor_17: float; sensor_18: float; sensor_19: float; sensor_20: float
    sensor_21: float


class PredictRequest(BaseModel):
    dataset: str = Field(..., examples=["FD001"])
    model: str | None = Field(None, description="Leave empty to use the model selected on validation data.")
    records: list[EngineRow] = Field(..., description="Every cycle of each engine's history, in C-MAPSS columns.")


class ExplainRequest(BaseModel):
    dataset: str
    engine_id: int
    records: list[EngineRow]


class EnginePrediction(BaseModel):
    engine_id: int
    last_cycle: int
    n_cycles: int
    predicted_rul: float
    rul_lower: float
    rul_upper: float
    at_cap: bool
    health_index: int
    risk_level: str
    true_rul: int | None = None


class PredictResponse(BaseModel):
    dataset: str
    model: str
    rul_cap: float
    interval_coverage: float
    interval_halfwidth: float
    engines: list[EnginePrediction]
    test_interval_coverage: float | None = None
    latency_ms: float | None = None


# --------------------------------------------------------------------------- error handling
def run(fn, *args, **kw):
    try:
        return fn(*args, **kw)
    except KeyError as e:
        raise HTTPException(404, str(e).strip("'\""))
    except service.InputError as e:
        raise HTTPException(422, str(e))
    except FileNotFoundError as e:
        raise HTTPException(503, str(e))


# --------------------------------------------------------------------------- endpoints
@app.get("/health")
def health():
    ok = {fd: (config.MODELS_DIR / fd / "selected_model.json").exists() for fd in config.DATASETS}
    return {"status": "ok" if all(ok.values()) else "degraded", "models_found": ok,
            "raw_data_found": (config.RAW_DIR / "test_FD001.txt").exists()}


@app.get("/datasets")
def datasets():
    return [run(service.dataset_info, fd) for fd in config.DATASETS]


@app.get("/datasets/{fd}")
def dataset(fd: str):
    return run(service.dataset_info, fd)


@app.get("/datasets/{fd}/engines")
def test_engines(fd: str):
    """Official test engines available for the demo (id + number of cycles)."""
    fd = run(service.check_dataset, fd)
    test, _ = run(service.raw_test, fd)
    n = test.groupby("engine_id").cycle.max()
    return [{"engine_id": int(i), "n_cycles": int(c)} for i, c in n.items()]


@app.get("/datasets/{fd}/engines/{engine_id}")
def test_engine(fd: str, engine_id: int):
    """Raw history of one official test engine, plus its true RUL (demonstration only)."""
    fd = run(service.check_dataset, fd)
    test, rul = run(service.raw_test, fd)
    one = test[test.engine_id == engine_id]
    if one.empty:
        raise HTTPException(404, f"Engine {engine_id} not in {fd} test set.")
    return {"engine_id": engine_id, "true_rul": int(rul.loc[rul.engine_id == engine_id, "true_rul"].iloc[0]),
            "records": one.to_dict(orient="records")}


@app.get("/datasets/{fd}/fleet", response_model=PredictResponse)
def fleet(fd: str):
    """Predictions for every official test engine, with true RUL and the interval coverage reached."""
    t = time.perf_counter()
    res = run(service.fleet, fd)
    res["latency_ms"] = round((time.perf_counter() - t) * 1000, 1)
    return res


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    t = time.perf_counter()
    df = run(service.records_to_frame, [r.model_dump() for r in req.records])
    res = run(service.predict, req.dataset, df, req.model)
    res["latency_ms"] = round((time.perf_counter() - t) * 1000, 1)
    return res


@app.post("/predict/file", response_model=PredictResponse)
async def predict_file(dataset: str = Form(...), file: UploadFile = File(...), model: str | None = Form(None)):
    """Upload a NASA-format .txt (26 columns, no header) or a CSV with C-MAPSS column names."""
    t = time.perf_counter()
    df = run(service.parse_upload, await file.read())
    res = run(service.predict, dataset, df, model or None)
    res["latency_ms"] = round((time.perf_counter() - t) * 1000, 1)
    return res


@app.post("/explain")
def explain(req: ExplainRequest):
    df = run(service.records_to_frame, [r.model_dump() for r in req.records])
    return run(service.explain, req.dataset, df, req.engine_id)


@app.get("/reports/comparison")
def comparison():
    return run(service.comparison)


@app.get("/reports/{fd}/feature-importance")
def feature_importance(fd: str):
    """Global importance of the selected tree model (from training)."""
    fd = run(service.check_dataset, fd)
    sel = service.selected_model(fd)["selected_model"]
    name = {"Random Forest": "rf", "XGBoost": "xgb"}.get(sel)
    if not name:
        raise HTTPException(404, f"No feature-importance file for {sel}.")
    s = pd.read_csv(config.REPORTS_DIR / "model_results" / fd / f"{name}_feature_importance.csv", index_col=0).iloc[:, 0]
    return {"dataset": fd, "model": sel,
            "importance": [{"feature": k, "importance": float(v)} for k, v in s.sort_values(ascending=False).items()]}
