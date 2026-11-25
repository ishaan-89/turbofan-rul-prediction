"""Model Comparison — all 16 trained models (4 datasets × 4 algorithms) on validation and test."""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import common as c
import streamlit as st

c.setup("Model Comparison", "📊")
try:
    df = pd.DataFrame(c.api_get("/reports/comparison"))
except RuntimeError as e:
    c.show_error(e)

st.markdown("Each dataset was modelled separately with 4 algorithms. The ★ model was chosen **on validation data only**, "
            "then evaluated once on the official NASA test set.")

metric = st.radio("Metric", ["Val RMSE (≤cap)", "Test RMSE (true ≤100, common)", "Test RMSE"], horizontal=True,
                  captions=["validation, engines near failure", "test, same yardstick for all datasets", "test, all engines"])
fig = px.bar(df, x="Dataset", y=metric, color="Model", barmode="group", text_auto=".1f",
             color_discrete_map=c.MODEL_COLORS, category_orders={"Model": c.MODELS})
fig.update_yaxes(title=f"{metric} (cycles, lower is better)")
st.plotly_chart(c.style_fig(fig, 430), use_container_width=True)
if metric == "Test RMSE":
    st.caption("FD002/FD004 overall test RMSE is high because many of their test engines have far more life left than the cap; "
               "a capped model correctly predicts 'healthy' for them without an exact number.")

a, b = st.columns([1.1, 1])
with a:
    st.subheader("Selected models")
    sel = df[df.Selected == "★"][["Dataset", "Model", "Val RMSE (≤cap)", "Test RMSE (true ≤cap)", "Test RMSE (true ≤100, common)",
                                   "Test RMSE", "Test MAE", "Test NASA", "Test bias", "Late > 10 (%)"]]
    st.dataframe(sel.round(2), hide_index=True, use_container_width=True)
    st.caption("Bias = mean(predicted − true). Negative = early (safe side). NASA score penalises late predictions more.")
with b:
    sc = px.scatter(df, x="Val RMSE (≤cap)", y="Test RMSE (true ≤cap)", color="Model", symbol="Dataset",
                    color_discrete_map=c.MODEL_COLORS, hover_data=["Dataset", "Model"])
    lo, hi = 7, 22
    sc.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(color="#FFFFFF", dash="dash"), name="test = validation"))
    st.plotly_chart(c.style_fig(sc, 360, "Did validation predict test error?"), use_container_width=True)

st.subheader("All 16 models")
show = df[["Dataset", "Model", "Selected", "Val RMSE (≤cap)", "Test RMSE (true ≤cap)", "Test RMSE (true ≤100, common)", "Test RMSE",
           "Test MAE", "Test NASA", "Model size (MB)", "CPU inference (ms)"]]
st.dataframe(show.round(2), hide_index=True, use_container_width=True, height=600)
