# RUL Prediction — FastAPI backend + Streamlit dashboard

Research prototype on simulated NASA C-MAPSS data. **Not certified aviation software.**

```
Streamlit dashboard (port 8501)  --HTTP-->  FastAPI (port 8000)  -->  saved models in models/<FD>/
```
The dashboard never loads models itself; every number comes from the API, which uses exactly the
saved preprocessing, feature order and model that the notebooks produced.

## 1. Folder layout
```
<any folder>/
├── CMAPSSData/                    raw NASA files (train_FD001.txt, test_FD001.txt, RUL_FD001.txt, ...)
└── Predictive_Maintenance_RUL/    downloaded from Google Drive (src/, models/, reports/, ...)
    ├── api/            <- from this zip
    ├── dashboard/      <- from this zip
    ├── tests/          <- from this zip
    ├── .streamlit/     <- from this zip (dark navy theme)
    ├── pytest.ini, requirements_app.txt, start_api.bat, start_dashboard.bat, run_tests.bat
```
If the raw folder has another name or place, set it before starting:
`set CMAPSS_RAW_DIR=C:\path\to\CMAPSSData` (PowerShell: `$env:CMAPSS_RAW_DIR="C:\path\to\CMAPSSData"`).

## 2. Install (once)
Open a terminal **inside `Predictive_Maintenance_RUL`**:
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements_app.txt
```

## 3. Run
Terminal 1: `start_api.bat` (or `uvicorn api.main:app --port 8000`). API docs at http://127.0.0.1:8000/docs
Terminal 2: `start_dashboard.bat` (or `streamlit run dashboard/Home.py`). Dashboard at http://localhost:8501

## 4. Test
`run_tests.bat` (or `pytest -q`). 30 tests: the API reproduces the notebooks' official test predictions,
bad input is rejected (missing columns, NaN, duplicates, too-short history, unknown operating regime),
predictions are causal, SHAP values add up to the prediction, the uncertainty range uses validation data only.

## API endpoints
| Method | Path | What it does |
|---|---|---|
| GET | `/health` | models and raw data found? |
| GET | `/datasets`, `/datasets/{fd}` | selected model, cap, window, test result |
| GET | `/datasets/{fd}/engines` | official test engines |
| GET | `/datasets/{fd}/engines/{id}` | one engine's raw history + true RUL (demo) |
| GET | `/datasets/{fd}/fleet` | predictions for all test engines + interval coverage |
| POST | `/predict` | JSON rows → RUL, 90% range, Health Index, risk |
| POST | `/predict/file` | upload NASA `.txt` or CSV → same |
| POST | `/explain` | SHAP contributions for one engine (tree models) |
| GET | `/reports/comparison` | all 16 models, validation + test |
| GET | `/reports/{fd}/feature-importance` | global importance of the selected model |

## Definitions used by the app
- **Health Index** (custom project metric) = 100 × min(predicted RUL, cap) / cap.
- **Risk levels** (demonstration thresholds): Critical < 20, High 20–49, Moderate 50–89, Low ≥ 90 cycles. Edit in `api/config.py`.
- **90% range**: split-conformal half-width from the selected model's validation errors (capped target).
  Coverage reached on the official test sets: FD001 93%, FD002 88%, FD003 84%, FD004 92%.
