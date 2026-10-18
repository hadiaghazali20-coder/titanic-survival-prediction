"""Heart Disease Predictor (Streamlit). Loads heart_disease_model.pkl, the complete
scikit-learn pipeline (pre-processing + Logistic Regression) that uses exactly 4 inputs."""
import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️")

@st.cache_resource
def load_model():
    with open(Path(__file__).parent / "heart_disease_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()
FEATURES = ["age", "sex", "cp", "thalach"]

CHEST_PAIN = {"Typical angina": 1, "Atypical angina": 2, "Non-anginal pain": 3, "Asymptomatic": 4}

st.title("❤️ Heart Disease Predictor")
st.write("Enter four details about a patient. A Logistic Regression model trained on the UCI Heart Disease (Cleveland) data predicts whether heart disease is present.")

age = st.number_input("Age (years)", min_value=20, max_value=100, value=55)
sex_label = st.selectbox("Sex", ["Male", "Female"])
cp_label = st.selectbox("Chest pain type", list(CHEST_PAIN))
thalach = st.number_input("Maximum heart rate achieved (bpm)", min_value=60, max_value=220, value=150)

if st.button("Predict", type="primary"):
    row = pd.DataFrame({"age": [age], "sex": [1 if sex_label == "Male" else 0],
                        "cp": [CHEST_PAIN[cp_label]], "thalach": [thalach]})[FEATURES]
    label = int(model.predict(row)[0])
    probability = float(model.predict_proba(row)[0, 1])
    if label == 1:
        st.error(f"Heart disease likely (probability {probability:.0%})")
    else:
        st.success(f"Heart disease unlikely (probability of disease {probability:.0%})")

st.caption("Student project, not a medical tool. Do not use it for real health decisions.")
