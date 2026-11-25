"""Fleet & Upload — rank many engines by risk: the official test fleet, or a file you upload."""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import common as c
import streamlit as st

c.setup("Fleet & Upload", "📋")
tab_fleet, tab_upload = st.tabs(["Official test fleet", "Upload your own file"])


def results_table(df: pd.DataFrame, with_truth: bool):
    cols = ["engine_id", "n_cycles", "predicted_rul", "rul_lower", "rul_upper", "health_index", "risk_level"] + (["true_rul"] if with_truth else [])
    show = df.sort_values(["risk_level", "predicted_rul"])[cols]
    st.dataframe(show, hide_index=True, use_container_width=True, height=420,
                 column_config={"health_index": st.column_config.ProgressColumn("Health Index", min_value=0, max_value=100, format="%d"),
                                "predicted_rul": st.column_config.NumberColumn("Predicted RUL", format="%.1f"),
                                "rul_lower": "90% low", "rul_upper": "90% high", "risk_level": "Risk",
                                "n_cycles": "Cycles seen", "true_rul": "True RUL (demo)"})
    st.download_button("⬇️ Download predictions (CSV)", show.to_csv(index=False).encode(), "rul_predictions.csv", "text/csv")


def risk_bar(df):
    counts = df.risk_level.value_counts().reindex(c.RISK_ORDER, fill_value=0)
    fig = go.Figure(go.Bar(x=counts.index, y=counts.values, marker_color=[c.RISK_COLORS[r] for r in counts.index],
                           text=counts.values, textposition="outside"))
    return c.style_fig(fig, 320, "Engines per risk level")


with tab_fleet:
    fd = st.selectbox("Dataset", c.DATASETS, key="fleet_fd")
    with st.spinner("Predicting every test engine..."):
        try:
            res = c.api_get(f"/datasets/{fd}/fleet")
        except RuntimeError as e:
            c.show_error(e)
    df = c.engines_frame(res)
    k = st.columns(4)
    c.kpi(k[0], "Engines", f"{len(df)}", f"model: {res['model']}")
    c.kpi(k[1], "Critical (< 20 cycles)", f"{(df.risk_level == 'Critical').sum()}", "inspect first", c.RED)
    c.kpi(k[2], "Mean Health Index", f"{df.health_index.mean():.0f}", "0 failure · 100 healthy", c.GREEN)
    c.kpi(k[3], "True RUL inside 90% range", f"{100 * res['test_interval_coverage']:.0f}%",
          f"target 90% · range ±{res['interval_halfwidth']:.1f} cycles", c.BLUE)
    a, b = st.columns(2)
    a.plotly_chart(risk_bar(df), use_container_width=True)
    cap = res["rul_cap"]
    sc = px.scatter(df, x="true_rul", y="predicted_rul", color="risk_level", color_discrete_map=c.RISK_COLORS,
                    category_orders={"risk_level": c.RISK_ORDER}, hover_data=["engine_id"])
    sc.add_trace(go.Scatter(x=[0, cap], y=[0, cap], mode="lines", line=dict(color="#FFFFFF", dash="dash"), name="perfect (up to cap)"))
    sc.update_xaxes(title="True RUL"); sc.update_yaxes(title="Predicted RUL"); sc.update_layout(legend_title_text="")
    b.plotly_chart(c.style_fig(sc, 320, "Predicted vs true RUL"), use_container_width=True)
    results_table(df, with_truth=True)

with tab_upload:
    st.markdown("Uploading your own engines now has its own page, with a template, an example, per-engine trend charts "
                "and SHAP explanations.")
    st.page_link("pages/1_Predict_Your_Engine.py", label="Go to Predict Your Engine", icon="🔮")
