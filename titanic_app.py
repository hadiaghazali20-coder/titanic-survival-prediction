"""Titanic Passenger Survival Predictor (Streamlit).
Adapted from the instructor's streamlit-app (github.com/haqnawaz99/ml-teaching-notebooks).
Loads titanic_model.pkl, which holds the trained SVC and the four label encoders."""
import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢")

@st.cache_resource
def load_saved():
    with open(Path(__file__).parent / "titanic_model.pkl", "rb") as f:
        return pickle.load(f)

saved = load_saved()
model, encoders, features = saved["model"], saved["encoders"], saved["features"]

st.title("🚢 Titanic Passenger Survival Predictor")
st.write("Choose a passenger's details. A Support Vector Classifier trained on the 100-row teaching dataset predicts survival.")

pclass = st.selectbox("Passenger class (PClass)", ["First", "Second", "Third"], index=2)
gender = st.selectbox("Gender", ["Male", "Female"])
sibling = st.selectbox("Siblings/spouses aboard", ["Zero", "One", "Two", "Three"])
embarked = st.selectbox("Port of embarkation", ["Southampton", "Cherbourg", "Queenstown"])

if st.button("Predict survival", type="primary"):
    row = pd.DataFrame({"PClass": [pclass], "Gender": [gender], "Sibling": [sibling], "Embarked": [embarked]})
    for name in features:
        row[name] = encoders[name].transform(row[name])
    prediction = int(model.predict(row[features])[0])
    if prediction == 1:
        st.success("SURVIVED")
    else:
        st.error("NOT SURVIVED")

st.caption("Student teaching project based on a small simplified dataset. Not a historical analysis.")
