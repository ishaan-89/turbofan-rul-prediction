"""Predict Your Engine — upload (or paste) the sensor history of your own engines and get their remaining life.

Input: C-MAPSS layout (engine_id, cycle, 3 operating settings, 21 sensors), as a NASA .txt or a CSV.
The file is validated by the same API code used everywhere else; nothing is guessed or filled in.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go

import common as c
import streamlit as st
from api.service import InputError, parse_upload
from src.data_loader import COLUMNS

c.setup("Predict Your Engine", "🔮")
st.markdown("Upload the sensor history of **your own engines** and the model estimates how many operating cycles "
            "each one has left before failure, with an uncertainty range, a Health Index and a risk level.")

SS = st.session_state
SS.setdefault("pye_source", None)          # "upload" | "paste" | "example"
SS.setdefault("pye_truth", {})             # true RUL, only for NASA example engines


def _set_source(src):
    SS.pye_source = src
    if src != "example":
        SS.pye_truth = {}


# --------------------------------------------------------------------------- 1. input
st.subheader("1 · Add engine data")
tab_up, tab_paste, tab_ex = st.tabs(["📁 Upload a file", "📋 Paste data", "🧪 Try an example"])

with tab_up:
    st.file_uploader("NASA-format .txt or CSV, one row per engine per cycle", type=["txt", "csv"],
                     key="pye_file", on_change=_set_source, args=("upload",))
    t1, t2, _ = st.columns([1, 1, 2])
    t1.download_button("⬇️ Empty CSV template", (",".join(COLUMNS) + "\n").encode(), "engine_template.csv", "text/csv",
                       use_container_width=True)
    try:
        sample = pd.concat([pd.DataFrame(c.api_get(f"/datasets/FD001/engines/{i}")["records"]) for i in (1, 2, 3)])
        t2.download_button("⬇️ Sample file (3 engines)", sample[COLUMNS].to_csv(index=False).encode(),
                           "sample_engines.csv", "text/csv", use_container_width=True)
    except RuntimeError:
        pass

with tab_paste:
    st.text_area("Paste rows (with or without the header line; spaces, commas or semicolons)", key="pye_text", height=180,
                 placeholder="engine_id,cycle,op_setting_1,op_setting_2,op_setting_3,sensor_1,...,sensor_21\n"
                             "1,1,-0.0007,-0.0004,100.0,518.67,641.82,...\n1,2,0.0019,-0.0003,100.0,518.67,642.15,...")
    st.button("Predict pasted data", type="primary", on_click=_set_source, args=("paste",))

with tab_ex:
    st.markdown("Load engines from NASA's official test set as if they were your own file. "
                "Their true remaining life is known, so you can compare it with the prediction.")
    e1, e2, e3 = st.columns([1, 1, 1])
    ex_fd = e1.selectbox("Example from", c.DATASETS, key="pye_ex_fd")
    ex_n = e2.slider("Number of engines", 1, 10, 5, key="pye_ex_n")
    if e3.button("Load example engines", type="primary", use_container_width=True):
        try:
            ids = [e["engine_id"] for e in c.api_get(f"/datasets/{ex_fd}/engines")][:ex_n]
            engs = [c.api_get(f"/datasets/{ex_fd}/engines/{i}") for i in ids]
            SS.pye_example = pd.concat([pd.DataFrame(e["records"]) for e in engs])[COLUMNS].to_csv(index=False).encode()
            SS.pye_source = "example"
            SS.pye_truth = {e["engine_id"]: e["true_rul"] for e in engs}
        except RuntimeError as e:
            st.error(str(e))

with st.expander("What data format does the model need?"):
    st.markdown(f"""
Each row is **one engine at one operating cycle**, with these **26 columns in this order**:

`engine_id, cycle, op_setting_1, op_setting_2, op_setting_3, sensor_1, …, sensor_21`

- Include **every cycle from the start of the engine's life (or monitoring) up to now**; the model looks at
  trends over the last 30 cycles (200 for six-condition data).
- Many engines can be in one file; they are told apart by `engine_id`.
- Accepted: NASA `.txt` (space-separated, no header) or CSV/Excel export (comma or semicolon, with or without the header line).
- Rows with missing or non-numeric values, duplicate cycles, or unknown operating conditions are **rejected with a message**;
  the app never fills in or guesses values.

**Important:** the models learned the 21 sensors of the NASA C-MAPSS turbofan simulation. They can only predict for
engines measured with the same sensors and units. Data from a different kind of engine or machine would need a model trained on that data.
""")

content, name = None, ""
if SS.pye_source == "upload" and SS.get("pye_file") is not None:
    content, name = SS.pye_file.getvalue(), SS.pye_file.name
elif SS.pye_source == "paste" and SS.get("pye_text", "").strip():
    content, name = SS.pye_text.encode(), "pasted data"
elif SS.pye_source == "example" and SS.get("pye_example"):
    content, name = SS.pye_example, f"NASA {SS.pye_ex_fd} example"

if content is None:
    st.info("Upload a file, paste rows, or load an example to get a prediction.")
    st.stop()


@st.cache_data(show_spinner=False, max_entries=20)
def parse(data: bytes) -> pd.DataFrame:
    return parse_upload(data)


@st.cache_data(show_spinner=False, max_entries=50)
def predict(fd: str, model: str, df: pd.DataFrame) -> dict:
    return c.api_post("/predict", {"dataset": fd, "model": model, "records": df.to_dict(orient="records")})


try:
    df = parse(content)
except InputError as e:
    st.error(f"**This data could not be used:** {e}")
    st.stop()

# --------------------------------------------------------------------------- 2. operating profile + model
multi = bool((df.op_setting_1.abs() > 1).any())         # six-condition data has op_setting_1 values up to ~42
st.subheader("2 · Check the data and choose the model")
k = st.columns(4)
c.kpi(k[0], "Source", name if len(name) < 28 else name[:25] + "…")
c.kpi(k[1], "Engines", f"{df.engine_id.nunique()}")
c.kpi(k[2], "Rows (engine-cycles)", f"{len(df):,}")
c.kpi(k[3], "Operating conditions", "several" if multi else "one", "detected from op_setting_1–3")
st.markdown("")

is_example = SS.pye_source == "example"
auto_fd = SS.pye_ex_fd if is_example else ("FD004" if multi else "FD003")
labels = {"auto": f"Auto: {auto_fd} model (recommended)", **{fd: f"{fd} model · {c.DATASET_NOTES[fd]}" for fd in c.DATASETS}}
m1, m2 = st.columns(2)
choice = m1.selectbox("Model family", list(labels), format_func=labels.get, key="pye_fd",
                      help="Auto picks the model trained on both fault modes for the operating conditions found in your data.")
fd = auto_fd if choice == "auto" else choice
if multi and fd in ("FD001", "FD003"):
    st.error(f"Your data contains several operating conditions, but {fd} was trained on one condition only. "
             "Choose FD002, FD004 or Auto.")
    st.stop()
try:
    info = c.api_get(f"/datasets/{fd}")
except RuntimeError as e:
    c.show_error(e)
model = m2.selectbox("Algorithm", c.MODELS, index=c.MODELS.index(info["selected_model"]), key=f"pye_model_{fd}",
                     help="The default is the algorithm that won on validation data for this family.")
if choice == "auto" and is_example:
    st.caption(f"NASA example engines from {fd}, so the **{fd}** model is used.")
elif choice == "auto":
    st.caption(f"Detected {'several operating conditions' if multi else 'one operating condition'}, so the "
               f"**{fd}** model is used: it was trained on both known fault modes (HPC and fan degradation), "
               "so it does not assume which part is wearing out.")

with st.spinner("Predicting…"):
    try:
        res = predict(fd, model, df)
    except RuntimeError as e:
        st.error(f"**Prediction failed:** {e}")
        st.stop()

cap = res["rul_cap"]
eng = c.engines_frame(res).drop(columns=["true_rul"], errors="ignore")
if SS.pye_truth and is_example:
    eng["true_rul"] = eng.engine_id.map(SS.pye_truth)

# --------------------------------------------------------------------------- 3. results for all engines
st.subheader("3 · Remaining life of every engine")
r = st.columns(4)
worst = eng.sort_values("predicted_rul").iloc[0]
c.kpi(r[0], "Engines predicted", f"{len(eng)}", f"{res['model']} · {res['latency_ms']:.0f} ms", c.BLUE)
c.kpi(r[1], "Critical (< 20 cycles)", f"{int((eng.risk_level == 'Critical').sum())}", "inspect first", c.RED)
c.kpi(r[2], "Lowest remaining life", f"{worst.predicted_rul:.0f}", f"engine {int(worst.engine_id)}", c.AMBER)
c.kpi(r[3], "Mean Health Index", f"{eng.health_index.mean():.0f}", "0 failure · 100 healthy", c.GREEN)
st.markdown("")

show = eng.sort_values(["risk_level", "predicted_rul"]).copy()
show["range"] = show.apply(lambda x: f"{x.rul_lower:.0f} – {x.rul_upper:.0f}", axis=1)
show["predicted"] = show.apply(lambda x: f"≥ {cap:.0f}" if x.at_cap else f"{x.predicted_rul:.0f}", axis=1)
cols = ["engine_id", "n_cycles", "predicted", "range", "health_index", "risk_level"] + (["true_rul"] if "true_rul" in show else [])
st.dataframe(show[cols], hide_index=True, use_container_width=True,
             column_config={"engine_id": "Engine", "n_cycles": "Cycles observed", "predicted": "Predicted RUL (cycles)",
                            "range": "90% range", "risk_level": "Risk",
                            "health_index": st.column_config.ProgressColumn("Health Index", min_value=0, max_value=100, format="%d"),
                            "true_rul": "True RUL (NASA example)"})
st.download_button("⬇️ Download predictions (CSV)", eng.drop(columns=["true_rul"], errors="ignore").to_csv(index=False).encode(),
                   "rul_predictions.csv", "text/csv")

# --------------------------------------------------------------------------- 4. one engine in detail
st.subheader("4 · Engine detail")
order = show.engine_id.tolist()
eid = st.selectbox("Engine", order, index=0, help="Sorted by risk: the most urgent engine comes first.")
p = eng[eng.engine_id == eid].iloc[0]
hist = df[df.engine_id == eid].sort_values("cycle")

d = st.columns(4)
c.kpi(d[0], "Predicted RUL (cycles)", f"≥ {cap:.0f}" if p.at_cap else f"{p.predicted_rul:.0f}",
      f"90% range {p.rul_lower:.0f} – {p.rul_upper:.0f}", c.BLUE)
c.kpi(d[1], "Health Index", f"{p.health_index}", "0 = failure · 100 = healthy", c.GREEN if p.health_index >= 50 else c.AMBER)
c.kpi(d[2], "Risk level", c.risk_badge(p.risk_level), "demonstration thresholds")
c.kpi(d[3], "History used", f"{int(p.n_cycles)} cycles", f"last cycle {int(p.last_cycle)}", c.MUTED)
if p.at_cap:
    st.info(f"The prediction is at the cap ({cap:.0f}): this engine has **at least {cap:.0f} cycles** left and looks healthy; "
            "the model does not give an exact number that far from failure.")
st.markdown("")

g, t = st.columns([1, 2])
with g:
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=int(p.health_index), number={"suffix": " / 100"},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": c.BLUE},
               "steps": [{"range": [0, 100 * 20 / cap], "color": "#4A1F2A"},
                         {"range": [100 * 20 / cap, 100 * 50 / cap], "color": "#4A3A1A"},
                         {"range": [100 * 50 / cap, 100], "color": "#16324A"}]}))
    st.plotly_chart(c.style_fig(fig, 300, "Health Index"), use_container_width=True)
with t:
    last = int(hist.cycle.max())
    first = int(hist.cycle.min())
    cuts = sorted(set(np.linspace(min(first + 30, last), last, num=min(20, last - first + 1)).astype(int)))
    parts = []
    for cut in cuts:
        part = hist[hist.cycle <= cut].copy()
        part["engine_id"] = cut                          # each cut point becomes its own pseudo-engine
        parts.append(part)
    try:
        tr = predict(fd, model, pd.concat(parts))
        tdf = pd.DataFrame(tr["engines"]).rename(columns={"engine_id": "cycle"})
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=tdf.cycle, y=tdf.rul_upper, line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=tdf.cycle, y=tdf.rul_lower, fill="tonexty", fillcolor="rgba(59,163,255,0.18)",
                                 line=dict(width=0), name="90% range"))
        fig.add_trace(go.Scatter(x=tdf.cycle, y=tdf.predicted_rul, mode="lines+markers", line=dict(color=c.BLUE, width=3),
                                 name="Predicted RUL"))
        if "true_rul" in eng and pd.notna(p.get("true_rul")):
            fig.add_trace(go.Scatter(x=tdf.cycle, y=p.true_rul + last - tdf.cycle, mode="lines",
                                     line=dict(color="#FFFFFF", dash="dash"), name="True RUL"))
        fig.add_hline(y=cap, line=dict(color=c.AMBER, dash="dot"), annotation_text=f"cap {cap:.0f}", annotation_font_color=c.AMBER)
        fig.update_xaxes(title="cycle (history used so far)"); fig.update_yaxes(title="RUL (cycles)")
        st.plotly_chart(c.style_fig(fig, 300, "Prediction as more history arrives"), use_container_width=True)
    except RuntimeError as e:
        st.caption(f"Trend chart unavailable for this engine: {e}")

x1, x2 = st.columns(2)
with x1:
    if model == info["selected_model"] and model in ("Random Forest", "XGBoost"):
        try:
            ex = c.api_post("/explain", {"dataset": fd, "engine_id": int(eid), "records": hist.to_dict(orient="records")})
            bs = pd.DataFrame(ex["by_sensor"]).head(10).iloc[::-1]
            fig = go.Figure(go.Bar(x=bs.shap, y=bs.sensor, orientation="h",
                                   marker_color=[c.GREEN if v > 0 else c.RED for v in bs.shap],
                                   text=[f"{v:+.1f}" for v in bs.shap], textposition="outside"))
            fig.update_xaxes(title="effect on predicted RUL (cycles)")
            st.plotly_chart(c.style_fig(fig, 360, "Why this prediction? (SHAP, top 10 sensors)"), use_container_width=True)
            st.caption("Red = pushes the prediction toward failure; green = toward more remaining life.")
        except RuntimeError as e:
            st.caption(f"Explanation unavailable: {e}")
    else:
        st.caption("Sensor-level explanations (SHAP) are shown for the recommended tree model of each family.")
with x2:
    sensors = info["base_features"]
    chosen = st.multiselect("Sensors to plot", sensors, default=sensors[:4], key="pye_sensors")
    if chosen:
        fig = go.Figure()
        for s in chosen:
            v = hist[s]
            fig.add_trace(go.Scatter(x=hist.cycle, y=(v - v.mean()) / (v.std() or 1), mode="lines", name=s))
        fig.update_yaxes(title="z-score within this engine"); fig.update_xaxes(title="cycle")
        st.plotly_chart(c.style_fig(fig, 360, "Sensor history"), use_container_width=True)

st.caption("Research prototype trained on simulated NASA C-MAPSS data. Not certified aviation software; "
           "not for real maintenance decisions.")
