"""
Tests for the RUL API. Run from the project folder:   pytest -q tests
Needs the trained models (models/), reports/ and the raw NASA files (CMAPSS_RAW_DIR).
"""
import json

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api import config, service
from api.main import app
from src.data_loader import COLUMNS

client = TestClient(app)
DATASETS = config.DATASETS


def engine_records(fd="FD001", engine_id=1):
    test, _ = service.raw_test(fd)
    return test[test.engine_id == engine_id].to_dict(orient="records")


# ----------------------------------------------------------------- API basics
def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_datasets_list_has_selected_models():
    r = client.get("/datasets")
    assert r.status_code == 200
    assert [d["dataset"] for d in r.json()] == DATASETS
    for d in r.json():
        assert d["selected_model"] in service.MODEL_NAMES


def test_unknown_dataset_is_404():
    assert client.get("/datasets/FD009").status_code == 404


# ----------------------------------------------------------------- predictions
@pytest.mark.parametrize("fd", DATASETS)
def test_api_reproduces_notebook_test_predictions(fd):
    """API predictions must equal the notebook's official test predictions (same saved pipeline)."""
    res = service.fleet(fd)
    ref = pd.read_csv(config.REPORTS_DIR / "model_results" / fd / "test_predictions.csv").set_index("engine_id")
    pred = pd.DataFrame(res["engines"]).set_index("engine_id")["predicted_rul"]
    assert len(pred) == len(ref)
    assert np.abs(pred - ref.loc[pred.index, res["model"]]).max() < 0.06      # API rounds to 0.1


def test_predict_response_fields_and_ranges():
    r = client.post("/predict", json={"dataset": "FD001", "records": engine_records()})
    assert r.status_code == 200
    body = r.json()
    e = body["engines"][0]
    assert body["model"] == service.selected_model("FD001")["selected_model"]
    assert 0 <= e["rul_lower"] <= e["predicted_rul"] <= e["rul_upper"] <= body["rul_cap"]
    assert 0 <= e["health_index"] <= 100
    assert e["risk_level"] in {"Critical", "High", "Moderate", "Low"}


@pytest.mark.parametrize("model", service.MODEL_NAMES)
def test_every_model_can_predict(model):
    r = client.post("/predict", json={"dataset": "FD003", "model": model, "records": engine_records("FD003", 1)})
    assert r.status_code == 200, r.text


def test_row_order_does_not_matter():
    recs = engine_records()
    a = client.post("/predict", json={"dataset": "FD001", "records": recs}).json()
    b = client.post("/predict", json={"dataset": "FD001", "records": recs[::-1]}).json()
    assert a["engines"][0]["predicted_rul"] == b["engines"][0]["predicted_rul"]


def test_future_rows_do_not_change_past_prediction():
    """Causality: predicting at cycle 20 with or without later cycles gives the same answer."""
    full = pd.DataFrame(engine_records())
    early = full[full.cycle <= 20]
    a = service.predict("FD001", service.records_to_frame(early))["engines"][0]["predicted_rul"]
    p = service.predictor("FD001", service.selected_model("FD001")["selected_model"])
    b = float(p.predict_at(full, pd.DataFrame({"engine_id": [1], "cut_cycle": [20]}))[0])
    assert abs(a - b) < 0.06


def test_file_upload_nasa_txt_format():
    df = pd.DataFrame(engine_records("FD002", 3))
    txt = df[COLUMNS].to_csv(sep=" ", header=False, index=False).encode()
    r = client.post("/predict/file", data={"dataset": "FD002"}, files={"file": ("x.txt", txt)})
    assert r.status_code == 200, r.text
    assert r.json()["engines"][0]["engine_id"] == 3


@pytest.mark.parametrize("variant", ["excel_bom_csv", "csv_no_header", "semicolon_csv"])
def test_file_upload_common_spreadsheet_formats_match_txt(variant):
    df = pd.DataFrame(engine_records("FD001", 5))[COLUMNS]
    ref = client.post("/predict/file", data={"dataset": "FD001"},
                      files={"file": ("x.txt", df.to_csv(sep=" ", header=False, index=False).encode())}).json()
    body = {"excel_bom_csv": df.to_csv(index=False).encode("utf-8-sig"),
            "csv_no_header": df.to_csv(index=False, header=False).encode(),
            "semicolon_csv": df.to_csv(index=False, sep=";").encode()}[variant]
    r = client.post("/predict/file", data={"dataset": "FD001"}, files={"file": ("x.csv", body)})
    assert r.status_code == 200, r.text
    assert r.json()["engines"][0]["predicted_rul"] == ref["engines"][0]["predicted_rul"]


def test_empty_upload_rejected():
    r = client.post("/predict/file", data={"dataset": "FD001"}, files={"file": ("x.csv", b"  \n")})
    assert r.status_code == 422


# ----------------------------------------------------------------- bad input is rejected, never guessed
def test_missing_column_rejected():
    recs = engine_records()
    for r in recs:
        r.pop("sensor_11")
    assert client.post("/predict", json={"dataset": "FD001", "records": recs}).status_code == 422


def test_nan_rejected_by_service():
    df = pd.DataFrame(engine_records())
    df.loc[3, "sensor_4"] = np.nan
    with pytest.raises(service.InputError):
        service.records_to_frame(df)


def test_duplicate_cycles_rejected():
    recs = engine_records()
    assert client.post("/predict", json={"dataset": "FD001", "records": recs + recs[:1]}).status_code == 422


def test_too_short_history_rejected_for_lstm():
    recs = engine_records()[:5]
    r = client.post("/predict", json={"dataset": "FD001", "model": "LSTM", "records": recs})
    assert r.status_code == 422


def test_unknown_operating_regime_rejected():
    recs = engine_records("FD002", 1)
    for r in recs:
        r["op_setting_1"] = 999.0
    assert client.post("/predict", json={"dataset": "FD002", "records": recs}).status_code == 422


def test_unknown_model_is_404():
    r = client.post("/predict", json={"dataset": "FD001", "model": "SVM", "records": engine_records()})
    assert r.status_code == 404


# ----------------------------------------------------------------- derived outputs
def test_health_index_and_risk_rules():
    assert service.health_index(125, 125) == 100
    assert service.health_index(200, 125) == 100
    assert service.health_index(0, 125) == 0
    assert service.health_index(-5, 100) == 0
    assert service.risk_level(5) == "Critical"
    assert service.risk_level(20) == "High"
    assert service.risk_level(50) == "Moderate"
    assert service.risk_level(90) == "Low"


def test_interval_uses_validation_not_test():
    """The uncertainty half-width must come from validation residuals only."""
    fd = "FD001"
    model = service.selected_model(fd)["selected_model"]
    v = pd.read_csv(config.REPORTS_DIR / "model_results" / fd / "validation_predictions.csv")
    cap = service.predictor(fd, model).cap
    r = np.abs(np.minimum(v.true_rul, cap) - v[model])
    q = service.conformal_halfwidth(fd, model)
    assert (r <= q + 1e-9).mean() >= config.INTERVAL_COVERAGE


@pytest.mark.parametrize("fd", ["FD001", "FD004"])
def test_explain_adds_up_to_prediction(fd):
    test, _ = service.raw_test(fd)
    ex = service.explain(fd, test, 1)
    total = ex["base_value"] + sum(s["shap"] for s in ex["by_sensor"])
    pred = service.predict(fd, test[test.engine_id == 1])["engines"][0]["predicted_rul"]
    assert abs(total - pred) < 0.5


def test_selection_was_frozen_before_test():
    for fd in DATASETS:
        sel = json.loads((config.MODELS_DIR / fd / "selected_model.json").read_text())
        assert sel["test_set_seen_before_selection"] is False
