"""Interactive Streamlit demo for MCU-A bandgap trim prediction."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
MODEL_PATH = ARTIFACTS / "bandgap_trim_model.joblib"
METRICS_PATH = ARTIFACTS / "metrics.json"

st.set_page_config(page_title="TrimFuse AI | Bandgap trim", page_icon="🔧", layout="wide")
st.title("TrimFuse AI")
st.subheader("MCU-A bandgap predictive trim")
st.write(
    "Enter measurements available before trim. The model suggests a 6-bit starting code; "
    "a tester measurement must verify the result before any fuse operation."
)
st.warning(
    "Classroom demonstration on synthetic data only. Do not use this prediction to program a production device."
)

if not MODEL_PATH.exists() or not METRICS_PATH.exists():
    st.error("Model artifacts are missing. Run python train_model.py from the project folder first.")
    st.stop()

bundle = joblib.load(MODEL_PATH)
metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
model = bundle["model"]
features = bundle["feature_names"]
training_ranges = metrics["feature_training_ranges"]

with st.sidebar:
    st.header("Held-out wafer results")
    holdout = metrics["holdout_metrics"]
    st.metric("Within ±1 trim code", f"{holdout['within_1_code_accuracy']:.2%}")
    st.metric("Exact-code accuracy", f"{holdout['exact_code_accuracy']:.1%}")
    st.metric("Mean absolute error", f"{holdout['mae_codes']:.3f} codes")
    st.caption(f"Model: {bundle['model_name']}")
    st.caption("Validation keeps wafers separate between training and test.")

st.markdown("### Enter pre-trim measurements")
st.caption("Example values are seeded from the data dictionary. Replace them with one die's measurements.")

with st.form("trim_input"):
    left, middle, right = st.columns(3)
    values = {}
    defaults = {
        "bg_untrim_V": 1.19017,
        "pcm_vth_n_mV": 452.04,
        "pcm_vth_p_mV": -477.49,
        "pcm_rs_ohm_sq": 284.88,
        "pcm_tox_A": 20.435,
        "pcm_mobility_rel": 1.0356,
        "x_mm": -24.0,
        "y_mm": -88.0,
        "radius_mm": 91.21,
    }
    labels = {
        "bg_untrim_V": "Bandgap before trim (V)",
        "pcm_vth_n_mV": "NMOS Vth (mV)",
        "pcm_vth_p_mV": "PMOS Vth (mV)",
        "pcm_rs_ohm_sq": "Poly sheet resistance (ohm/sq)",
        "pcm_tox_A": "Gate oxide thickness (Å)",
        "pcm_mobility_rel": "Relative mobility",
        "x_mm": "Wafer x coordinate (mm)",
        "y_mm": "Wafer y coordinate (mm)",
        "radius_mm": "Wafer radius (mm)",
    }
    for index, feature in enumerate(features):
        column = (left, middle, right)[index % 3]
        with column:
            values[feature] = st.number_input(
                labels[feature], value=float(defaults[feature]), format="%.5f", key=feature
            )
    submitted = st.form_submit_button("Predict candidate code", type="primary")

if submitted:
    input_frame = pd.DataFrame([[values[name] for name in features]], columns=features)
    raw_code = float(model.predict(input_frame)[0])
    predicted_code = int(max(bundle["code_min"], min(bundle["code_max"], round(raw_code))))
    st.markdown("### Prediction")
    st.metric("Suggested bandgap trim code", f"{predicted_code} / 63")
    st.info(
        "Apply this only as a candidate starting point in a controlled test flow. "
        "Measure the trimmed output, verify the tolerance, and use the conventional search "
        "if verification fails. The app does not write fuses."
    )

    outside = []
    for feature, value in values.items():
        limits = training_ranges[feature]
        if value < limits["min"] or value > limits["max"]:
            outside.append(feature)
    if outside:
        st.warning(
            "These inputs fall outside the observed training range: "
            + ", ".join(outside)
            + ". Treat the prediction as out of distribution and prefer a tester search."
        )

with st.expander("How to read the demo"):
    st.write(
        "The model regresses a numeric trim code, rounds it to the legal 0–63 range, "
        "and shows the held-out wafer metrics. Within-one-code accuracy matches the "
        "predict-then-verify use case: a near miss can be corrected by a short local check. "
        "The metrics are measured on synthetic held-out wafers, not on a production tester."
    )
