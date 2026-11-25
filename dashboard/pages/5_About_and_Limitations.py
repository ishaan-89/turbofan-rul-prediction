"""About & Limitations — definitions, thresholds, uncertainty check and honest limits."""
import pandas as pd

import common as c
import streamlit as st

c.setup("About & Limitations", "ℹ️")

st.markdown(f"""
<div class="card"><b style="color:{c.AMBER}">Project</b><br>
Intelligent Predictive Maintenance Using Machine Learning for Remaining Useful Life (RUL) Estimation<br>
<span class="muted">Ishaan Jain (1024030884) · Thapar Institute of Engineering and Technology</span></div>
""", unsafe_allow_html=True)

a, b = st.columns(2)
with a:
    st.subheader("Definitions")
    st.markdown("""
- **RUL**: operating cycles left before failure.
- **RUL cap**: training targets were capped (125 cycles; 100 for FD004) because sensors barely change early in life.
  A prediction at the cap means *"at least this many cycles, healthy"*.
- **Health Index** (custom project metric): `100 × min(predicted RUL, cap) / cap`. It is a rescaled prediction,
  not a physical measurement.
- **90% range**: split-conformal interval from validation errors of the selected model.
""")
    st.subheader("Risk levels (demonstration thresholds)")
    st.markdown(" ".join(c.risk_badge(r) for r in c.RISK_ORDER), unsafe_allow_html=True)
    st.markdown("""
| Level | Predicted RUL |
|---|---|
| Critical | < 20 cycles |
| High | 20 – 49 cycles |
| Moderate | 50 – 89 cycles |
| Low | ≥ 90 cycles |

These are illustrative cut-offs for the demo, **not** airline maintenance rules.
""")
with b:
    st.subheader("Does the 90% range hold on the test set?")
    rows = []
    for fd in c.DATASETS:
        try:
            r = c.api_get(f"/datasets/{fd}/fleet")
            rows.append({"Dataset": fd, "Model": r["model"], "Range (± cycles)": r["interval_halfwidth"],
                         "True RUL inside": f"{100 * r['test_interval_coverage']:.1f}%"})
        except RuntimeError:
            pass
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.caption("Target is 90%. The validation engines were also used to choose the model, so the range can be slightly "
               "optimistic; FD002 and FD003 reach less than 90% on their test sets.")
    st.subheader("Limitations")
    st.markdown("""
- Trained and tested on **simulated** NASA C-MAPSS data, not real airline engines.
- Valid only for engines and operating regimes like those in training; unknown regimes are rejected.
- Validation sets are small (20–52 engines): differences under ~0.5 cycles are noise.
- Overall test error is much higher for FD002/FD004 because long-life engines sit above the cap.
- Research and teaching prototype. **Not certified aviation software; not for real maintenance decisions.**
""")

st.subheader("API")
if c.API_URL:
    st.markdown(f"Interactive API documentation: [{c.API_URL}/docs]({c.API_URL}/docs)")
else:
    st.markdown("The FastAPI backend runs inside this dashboard. To browse its interactive docs, run it locally "
                "with `uvicorn api.main:app --port 8000` and open http://127.0.0.1:8000/docs.")
