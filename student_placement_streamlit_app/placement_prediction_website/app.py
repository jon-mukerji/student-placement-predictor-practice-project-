"""Student Placement Predictor — Streamlit interface for a fitted sklearn model."""
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Placement Predictor", page_icon="🎓", layout="centered")

st.markdown("""
<style>
.block-container {max-width: 850px; padding-top: 2.5rem;}
.hero {padding: 1.6rem 1.8rem; border-radius: 18px; background: linear-gradient(110deg,#10283e,#215d78); color: white; margin-bottom: 1.4rem;}
.hero h1 {color: white; font-size: 2.15rem; margin: 0;}
.hero p {color: #d5e7ef; margin: 0.65rem 0 0;}
.stButton > button {width: 100%; border-radius: 9px; font-weight: 700;}
</style>
""", unsafe_allow_html=True)

st.markdown("""<div class="hero"><h1>🎓 Student Placement Predictor</h1>
<p>Explore a Logistic Regression model trained to classify placement outcomes using CGPA and IQ.</p></div>""", unsafe_allow_html=True)

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "placement_model.pkl"
SCALER_PATH = ROOT / "scaler.pkl"

@st.cache_resource(show_spinner=False)
def load_artifacts(model_path: str, scaler_path: str):
    # Only load trusted model files. Unpickling can execute arbitrary Python code.
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    if not hasattr(model, "predict") or not hasattr(model, "predict_proba"):
        raise ValueError("The model must support predict() and predict_proba().")
    if not hasattr(scaler, "transform"):
        raise ValueError("The scaler must support transform().")
    if getattr(model, "n_features_in_", 2) != 2 or getattr(scaler, "n_features_in_", 2) != 2:
        raise ValueError("This app expects a model and scaler fitted on exactly two features: cgpa and iq.")
    for obj, label in ((model, "Model"), (scaler, "Scaler")):
        names = getattr(obj, "feature_names_in_", None)
        if names is not None and list(names) != ["cgpa", "iq"]:
            raise ValueError(f"{label} feature names/order should be ['cgpa', 'iq'], got {list(names)}")
    return model, scaler

if not (MODEL_PATH.is_file() and SCALER_PATH.is_file()):
    st.warning("Your trained model files aren't in this folder yet.")
    st.write("In Jupyter, save **both** the fitted model and the fitted scaler:")
    st.code('''import pickle
with open("placement_model.pkl", "wb") as f:
    pickle.dump(clf, f)
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)''', language="python")
    st.info("Put `placement_model.pkl` and `scaler.pkl` next to `app.py`, then refresh this page. Make sure the model was trained on features transformed by that same scaler.")
    st.stop()

try:
    model, scaler = load_artifacts(str(MODEL_PATH), str(SCALER_PATH))
except Exception as e:
    st.error(f"Could not load compatible model files: {e}")
    st.caption("Check that both files were saved from the same training session, and use compatible scikit-learn versions.")
    st.stop()

st.subheader("Enter student details")
with st.form("predict_form"):
    left, right = st.columns(2)
    with left:
        cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, value=7.0, step=0.1, help="Assuming your dataset uses a 0–10 CGPA scale")
    with right:
        iq = st.number_input("IQ", min_value=0.0, max_value=250.0, value=120.0, step=1.0)
    submit = st.form_submit_button("Predict placement", type="primary")

if submit:
    raw = pd.DataFrame([[cgpa, iq]], columns=["cgpa", "iq"])
    try:
        scaled = scaler.transform(raw)
        prediction = int(model.predict(scaled)[0])
        classes = list(model.classes_)
        if 1 not in classes or 0 not in classes:
            raise ValueError(f"Unexpected target classes: {classes}; expected 0 and 1.")
        prob = model.predict_proba(scaled)[0]
        placed_probability = float(prob[classes.index(1)])
        if prediction == 1:
            st.success("Prediction: Likely placed")
        else:
            st.info("Prediction: Not placed")
        col1, col2 = st.columns(2)
        col1.metric("Model probability: placed", f"{placed_probability:.1%}")
        col2.metric("Model probability: not placed", f"{1-placed_probability:.1%}")
        st.progress(min(1.0, max(0.0, placed_probability)), text="Estimated probability of placement")
        with st.expander("See data passed to the model"):
            st.dataframe(raw, hide_index=True)
            st.dataframe(pd.DataFrame(scaled, columns=["cgpa (scaled)", "iq (scaled)"]), hide_index=True)
    except Exception as exc:
        st.error(f"Prediction could not be completed: {exc}")

st.divider()
st.caption("Educational demo only. Probabilities reflect your small training dataset and may not be calibrated or reliable for real admission or hiring decisions. Never upload untrusted .pkl files.")
