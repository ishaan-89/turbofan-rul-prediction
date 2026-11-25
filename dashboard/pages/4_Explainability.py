"""Explainability — why the model predicted what it did (SHAP), and which features matter overall."""
import pandas as pd
import plotly.graph_objects as go

import common as c
import streamlit as st

c.setup("Explainability", "🧠")
st.markdown("**SHAP** splits one prediction into contributions: the model starts from an average RUL (*base value*) "
            "and each feature pushes it **up** (more life left) or **down** (closer to failure).")

a, b = st.columns(2)
fd = a.selectbox("Dataset", c.DATASETS)
try:
    engines = c.api_get(f"/datasets/{fd}/engines")
except RuntimeError as e:
    c.show_error(e)
engine_id = b.selectbox("Official test engine", [e["engine_id"] for e in engines])

try:
    eng = c.api_get(f"/datasets/{fd}/engines/{engine_id}")
    with st.spinner("Computing SHAP values..."):
        ex = c.api_post("/explain", {"dataset": fd, "engine_id": engine_id, "records": eng["records"]})
except RuntimeError as e:
    c.show_error(e)

k = st.columns(4)
c.kpi(k[0], "Model", ex["model"], fd)
c.kpi(k[1], "Base value (average)", f"{ex['base_value']:.1f}", "cycles")
c.kpi(k[2], "Prediction", f"{ex['prediction']:.1f}", "base + all contributions", c.BLUE)
c.kpi(k[3], "True RUL (demo only)", f"{eng['true_rul']}", "cycles", c.MUTED)

left, right = st.columns(2)
with left:
    s = pd.DataFrame(ex["by_sensor"]).head(10).iloc[::-1]
    fig = go.Figure(go.Bar(x=s.shap, y=s.sensor, orientation="h", marker_color=[c.GREEN if v > 0 else c.RED for v in s.shap],
                           text=[f"{v:+.1f}" for v in s.shap], textposition="outside"))
    fig.update_xaxes(title="effect on predicted RUL (cycles)")
    st.plotly_chart(c.style_fig(fig, 420, "Contribution per sensor (all its features combined)"), use_container_width=True)
with right:
    f = pd.DataFrame(ex["top_features"]).iloc[::-1]
    fig = go.Figure(go.Bar(x=f.shap, y=f.feature, orientation="h", marker_color=[c.GREEN if v > 0 else c.RED for v in f.shap],
                           text=[f"{v:+.1f}" for v in f.shap], textposition="outside"))
    fig.update_xaxes(title="effect on predicted RUL (cycles)")
    st.plotly_chart(c.style_fig(fig, 420, "Top individual features"), use_container_width=True)
st.caption("Red = pushes the prediction toward failure; green = toward more remaining life. Feature names: "
           "`sensor_11_mean_30` = average of sensor 11 over the last 30 cycles; `_std_` = variability; `_slope_` = trend; "
           "`cycle` = how long the engine has run.")

st.subheader(f"Global importance: {fd}")
try:
    imp = pd.DataFrame(c.api_get(f"/reports/{fd}/feature-importance")["importance"]).head(15).iloc[::-1]
    fig = go.Figure(go.Bar(x=imp.importance, y=imp.feature, orientation="h", marker_color=c.BLUE))
    st.plotly_chart(c.style_fig(fig, 460, "Most used features across all training data"), use_container_width=True)
except RuntimeError as e:
    st.info(str(e))
