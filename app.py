from pathlib import Path
import streamlit as st
import pandas as pd
import joblib

BASE_DIR = Path(__file__).resolve().parent
model = joblib.load(BASE_DIR / "cirrhosis_model_pipeline.pkl")
feature_names = joblib.load(BASE_DIR / "feature_names.pkl")

st.title("Cirrhosis Prediction App")
st.write("Enter the patient details below:")

input_data = {}

for col in feature_names:
    if col == "Sex":
        input_data[col] = st.selectbox(col, ["M", "F"])
    elif col in ["Ascites", "Hepatomegaly", "Spiders", "Edema"]:
        input_data[col] = st.selectbox(col, ["N", "Y"])
    else:
        input_data[col] = st.number_input(col, value=0.0)

stage_map = {
    1: "Stage 1",
    2: "Stage 2",
    3: "Stage 3",
    4: "Stage 4"
}

if st.button("Predict"):
    df = pd.DataFrame([input_data])
    prediction = model.predict(df)[0]

    stage_map = {
        1: ("Stage 1", "Earliest stage / mild liver damage"),
        2: ("Stage 2", "Moderate scarring"),
        3: ("Stage 3", "Advanced scarring"),
        4: ("Stage 4", "End-stage cirrhosis / liver failure")
    }

    stage_label, stage_desc = stage_map.get(
        int(prediction),
        (f"Unknown ({prediction})", "No description available")
    )

    st.success(f"Predicted cirrhosis stage: {stage_label}")
    st.write(stage_desc)