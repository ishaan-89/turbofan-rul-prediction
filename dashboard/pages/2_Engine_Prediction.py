"""Engine Prediction — one engine: RUL, uncertainty range, Health Index, risk, and how the prediction evolved."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go

import common as c
import streamlit as st

c.setup("Engine Prediction", "🔧")

left, mid, right = st.columns([1, 1, 1.4])
fd = left.selectbox("Dataset", c.DATASETS, help="FD001 easiest → FD004 hardest")
try:
    info = c.api_get(f"/datasets/{fd}")
    engines = c.api_get(f"/datasets/{fd}/engines")
except RuntimeError as e:
    c.show_error(e)
ids = [e["engine_id"] for e in engines]
engine_id = mid.selectbox("Official test engine", ids, index=min(30, len(ids) - 1) if fd == "FD001" else 0)
model = right.selectbox("Model", c.MODELS, index=c.MODELS.index(info["selected_model"]),
                        help="The default is the model selected on validation data.")

try:
    eng = c.api_get(f"/datasets/{fd}/engines/{engine_id}")
    res = c.api_post("/predict", {"dataset": fd, "model": model, "records": eng["records"]})
except RuntimeError as e:
    c.show_error(e)
p = res["engines"][0]
cap = res["rul_cap"]

k1, k2, k3, k4 = st.columns(4)
rul_txt = f"≥ {cap:.0f}" if p["at_cap"] else f"{p['predicted_rul']:.0f}"
c.kpi(k1, "Predicted RUL (cycles)", rul_txt, f"90% range {p['rul_lower']:.0f} – {p['rul_upper']:.0f}", c.BLUE)
c.kpi(k2, "Health Index", f"{p['health_index']}", "0 = failure · 100 = healthy (≥ cap)", c.GREEN if p["health_index"] >= 50 else c.AMBER)
c.kpi(k3, "Risk level", c.risk_badge(p["risk_level"]), "demonstration thresholds")
c.kpi(k4, "True RUL (demo only)", f"{eng['true_rul']}", f"after {p['n_cycles']} observed cycles · {res['latency_ms']:.0f} ms", c.MUTED)
if p["at_cap"]:
    st.info(f"The model predicts the cap ({cap:.0f}). This means \"at least {cap:.0f} cycles left, healthy\", not an exact number.")

st.markdown("")
g, t = st.columns([1, 2])
with g:
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=p["health_index"], number={"suffix": " / 100"},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": c.BLUE},
               "steps": [{"range": [0, 100 * 20 / cap], "color": "#4A1F2A"}, {"range": [100 * 20 / cap, 100 * 50 / cap], "color": "#4A3A1A"},
                         {"range": [100 * 50 / cap, 100], "color": "#16324A"}]}))
    st.plotly_chart(c.style_fig(fig, 300, "Health Index"), use_container_width=True)

# ---- how the prediction evolves over the engine's life: re-predict at earlier cut points (one API call)
hist = pd.DataFrame(eng["records"])
last = int(hist.cycle.max())
cuts = sorted(set(np.linspace(min(31, last), last, num=min(20, last)).astype(int)))
traj_records = []
for cut in cuts:
    part = hist[hist.cycle <= cut].copy()
    part["engine_id"] = cut                     # each cut point becomes its own pseudo-engine
    traj_records += part.to_dict(orient="records")
with t:
    try:
        tr = c.api_post("/predict", {"dataset": fd, "model": model, "records": traj_records})
        tdf = pd.DataFrame(tr["engines"]).rename(columns={"engine_id": "cycle"})
        tdf["true"] = eng["true_rul"] + last - tdf.cycle      # true RUL at each earlier cut point
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=tdf.cycle, y=tdf.rul_upper, line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=tdf.cycle, y=tdf.rul_lower, fill="tonexty", fillcolor="rgba(59,163,255,0.18)", line=dict(width=0), name="90% range"))
        fig.add_trace(go.Scatter(x=tdf.cycle, y=tdf.predicted_rul, mode="lines+markers", line=dict(color=c.BLUE, width=3), name="Predicted RUL"))
        fig.add_trace(go.Scatter(x=tdf.cycle, y=tdf.true, mode="lines", line=dict(color="#FFFFFF", dash="dash"), name="True RUL"))
        fig.add_hline(y=cap, line=dict(color=c.AMBER, dash="dot"), annotation_text=f"cap {cap:.0f}", annotation_font_color=c.AMBER)
        fig.update_xaxes(title="cycle (history used so far)"); fig.update_yaxes(title="RUL (cycles)")
        st.plotly_chart(c.style_fig(fig, 300, "Prediction as more history arrives"), use_container_width=True)
    except RuntimeError as e:
        st.warning(f"Trajectory unavailable: {e}")

st.subheader("Sensor history")
sensors = info["base_features"]
chosen = st.multiselect("Sensors to plot (raw values)", sensors, default=sensors[:4])
if chosen:
    fig = go.Figure()
    for s in chosen:
        x = hist[s]
        fig.add_trace(go.Scatter(x=hist.cycle, y=(x - x.mean()) / (x.std() or 1), mode="lines", name=s))
    fig.update_yaxes(title="z-score within this engine"); fig.update_xaxes(title="cycle")
    st.plotly_chart(c.style_fig(fig, 340), use_container_width=True)
    st.caption("Each sensor is standardised within this engine only so they fit on one chart. "
               "For FD002/FD004 the jumps come from changing operating conditions, which the model removes with per-regime scaling.")
