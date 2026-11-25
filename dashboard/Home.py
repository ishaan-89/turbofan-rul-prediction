"""Home.py — Overview page. Run from the project folder:  streamlit run dashboard/Home.py"""
import common as c
import streamlit as st

c.setup("Engine Health Overview", "✈️")
st.markdown("Estimate how many operating cycles a turbofan engine has left before failure "
            "(**Remaining Useful Life, RUL**) from its sensor history, using models trained and selected on NASA C-MAPSS data.")

try:
    info = c.api_get("/datasets")
except RuntimeError as e:
    c.show_error(e)

st.subheader("Selected model per dataset")
cols = st.columns(4)
for col, d in zip(cols, info):
    t = d["test_evaluation"]["selected_model_test_metrics"] if d["test_evaluation"] else {}
    near = next((v for k, v in t.items() if k.startswith("RMSE (true")), None)
    with col:
        st.markdown(f"""<div class="card">
          <div style="color:{c.BLUE};font-weight:700;font-size:1.3rem">{d['dataset']}</div>
          <div class="muted" style="font-size:0.85rem">{c.DATASET_NOTES[d['dataset']]}</div>
          <div style="font-size:1.15rem;font-weight:700;margin-top:10px">{d['selected_model']}</div>
          <div style="margin-top:8px"><span class="muted">Test RMSE near failure</span><br>
               <span style="font-size:1.8rem;font-weight:700;color:{c.GREEN}">{near:.2f}</span> <span class="muted">cycles</span></div>
          <div class="muted" style="font-size:0.85rem;margin-top:6px">Validation RMSE {d['validation_rmse']:.2f} · RUL cap {d['rul_cap']:.0f}<br>
               {d['test_evaluation']['n_engines']} official test engines</div>
        </div>""", unsafe_allow_html=True)

st.subheader("How a prediction is made")
steps = [("1 · Sensor history", "All cycles of one engine so far: 3 settings + 21 sensors."),
         ("2 · Check & scale", "Bad input is rejected. The saved training scaler is applied (one per operating regime)."),
         ("3 · Build features", "Rolling mean, std and slope of 14 sensors, in the saved order."),
         ("4 · Selected model", "The model chosen on validation data predicts RUL in cycles."),
         ("5 · Interpret", "90% uncertainty range, Health Index (0–100) and a risk level.")]
for col, (h, txt) in zip(st.columns(5), steps):
    col.markdown(f'<div class="card" style="min-height:150px"><b style="color:{c.AMBER}">{h}</b><br>'
                 f'<span style="font-size:0.9rem">{txt}</span></div>', unsafe_allow_html=True)

st.subheader("Pages")
st.markdown("""
- **Engine Prediction**: pick any official test engine and see its RUL, uncertainty, Health Index and risk.
- **Fleet & Upload**: rank a whole fleet by risk, or upload your own engine file.
- **Model Comparison**: all 16 trained models (4 datasets × 4 algorithms) on validation and test.
- **Explainability**: which sensors pushed a prediction up or down (SHAP).
- **About & Limitations**: definitions, thresholds and what this prototype cannot claim.
""")
st.caption("Research prototype on simulated NASA data. Not certified aviation software; not for real maintenance decisions.")
