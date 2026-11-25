"""common.py — Shared look, API calls and small widgets for every dashboard page.

Two ways to reach the FastAPI backend:
  * RUL_API_URL set (e.g. http://127.0.0.1:8000) -> real HTTP calls to a separately running uvicorn server.
  * RUL_API_URL not set (default, used on Streamlit Community Cloud) -> the same FastAPI app runs inside
    this Streamlit process and is called through FastAPI's TestClient. Same endpoints, same code path.
"""
import os
import sys
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:                 # so `import api` works when Streamlit runs dashboard/Home.py
    sys.path.insert(0, str(ROOT))

API_URL = os.environ.get("RUL_API_URL", "").rstrip("/")


@st.cache_resource(show_spinner="Loading models…")
def _client():
    """In-process FastAPI client, created once per server (None when using a remote API)."""
    if API_URL:
        return None
    from fastapi.testclient import TestClient
    from api.main import app
    return TestClient(app)


def _request(method: str, path: str, timeout: int = 120, **kw):
    client = _client()
    if client is not None:
        return client.request(method, path, **kw)
    return requests.request(method, f"{API_URL}{path}", timeout=timeout, **kw)
DATASETS = ["FD001", "FD002", "FD003", "FD004"]
MODELS = ["Linear Regression", "Random Forest", "XGBoost", "LSTM"]

NAVY, CARD, CARD2 = "#0B1A2E", "#13284A", "#1B3560"
BLUE, AMBER, GREEN, RED, GRAY, MUTED = "#3BA3FF", "#FFB020", "#2ECC9A", "#E5484D", "#7A8BA6", "#A9B8CC"
MODEL_COLORS = {"Linear Regression": GRAY, "Random Forest": BLUE, "XGBoost": GREEN, "LSTM": AMBER}
RISK_COLORS = {"Critical": RED, "High": AMBER, "Moderate": "#F5D76E", "Low": GREEN}
RISK_ORDER = ["Critical", "High", "Moderate", "Low"]
DATASET_NOTES = {"FD001": "1 condition · 1 fault mode", "FD002": "6 conditions · 1 fault mode",
                 "FD003": "1 condition · 2 fault modes", "FD004": "6 conditions · 2 fault modes"}


def setup(title: str, icon: str = "🛠️"):
    st.set_page_config(page_title=f"{title} · RUL Dashboard", page_icon=icon, layout="wide")
    st.markdown(f"""
    <style>
      .block-container {{padding-top: 2rem; max-width: 1350px;}}
      .kpi {{background:{CARD}; border-radius:12px; padding:16px 18px; height:100%;}}
      .kpi .label {{color:{MUTED}; font-size:0.85rem;}}
      .kpi .value {{font-size:2rem; font-weight:700; line-height:1.2;}}
      .kpi .sub {{color:{MUTED}; font-size:0.8rem;}}
      .badge {{display:inline-block; padding:4px 14px; border-radius:999px; font-weight:700; color:{NAVY};}}
      .card {{background:{CARD}; border-radius:12px; padding:18px 20px; margin-bottom:12px;}}
      .muted {{color:{MUTED};}}
      div[data-testid="stMetric"] {{background:{CARD}; border-radius:12px; padding:12px 16px;}}
    </style>""", unsafe_allow_html=True)
    with st.sidebar:
        st.markdown("### ✈️ Turbofan RUL")
        st.caption("Predictive maintenance on NASA C-MAPSS (simulated data). Research prototype, not certified aviation software.")
        ok = api_ok()
        where = f"`{API_URL}`" if API_URL else "built-in (same process)"
        st.markdown(f"API: {'🟢 connected' if ok else '🔴 not reachable'}  \n{where}")
        st.markdown("[GitHub repository](https://github.com/ishaan-89/turbofan-rul-prediction)")
    st.title(f"{icon} {title}")
    if not ok:
        st.error(f"The FastAPI backend at {API_URL} is not reachable. Start it from the project folder with:\n\n"
                 "`uvicorn api.main:app --port 8000`\n\nor unset RUL_API_URL to run it inside the dashboard.")
        st.stop()


def api_ok() -> bool:
    try:
        r = _request("GET", "/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def _handle(r):
    if r.status_code < 400:
        return r.json()
    try:
        detail = r.json().get("detail", r.text)
    except ValueError:
        detail = r.text
    raise RuntimeError(f"API error {r.status_code}: {detail}")


@st.cache_data(ttl=600, show_spinner=False)
def api_get(path: str):
    return _handle(_request("GET", path, timeout=120))


def api_post(path: str, payload: dict):
    return _handle(_request("POST", path, timeout=300, json=payload))


def api_post_file(path: str, data: dict, filename: str, content: bytes):
    return _handle(_request("POST", path, timeout=300, data=data, files={"file": (filename, content)}))


def kpi(col, label: str, value: str, sub: str = "", color: str = "#FFFFFF"):
    col.markdown(f'<div class="kpi"><div class="label">{label}</div>'
                 f'<div class="value" style="color:{color}">{value}</div><div class="sub">{sub}</div></div>',
                 unsafe_allow_html=True)


def risk_badge(level: str) -> str:
    return f'<span class="badge" style="background:{RISK_COLORS[level]}">{level}</span>'


def style_fig(fig, height=380, title=None):
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      height=height, margin=dict(l=10, r=10, t=50 if title else 20, b=10), title=title,
                      font=dict(color="#FFFFFF"), legend=dict(orientation="h", yanchor="top", y=-0.25))
    fig.update_xaxes(gridcolor="#29426A", zerolinecolor="#29426A")
    fig.update_yaxes(gridcolor="#29426A", zerolinecolor="#29426A")
    return fig


def engines_frame(res: dict) -> pd.DataFrame:
    df = pd.DataFrame(res["engines"])
    df["risk_level"] = pd.Categorical(df["risk_level"], RISK_ORDER, ordered=True)
    return df


def show_error(e: Exception):
    st.error(str(e))
    st.stop()
