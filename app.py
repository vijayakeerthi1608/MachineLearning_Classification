import os
import subprocess
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model" / "trained_models.joblib"
TRAIN_SCRIPT = ROOT / "model" / "train_models.py"
SAMPLE_DATA_PATH = ROOT / "test_data.csv"

st.set_page_config(page_title="Dry Bean Classifier", page_icon="🌱", layout="wide")


def ensure_models_exist() -> None:
    if MODEL_PATH.exists():
        return

    st.info("Training models for the first time. This may take a few seconds...")
    try:
        subprocess.run([sys.executable, str(TRAIN_SCRIPT)], check=True, cwd=str(ROOT), text=True)
    except subprocess.CalledProcessError as exc:
        st.error(f"Training failed: {exc}")
        st.stop()


ensure_models_exist()
artifact = joblib.load(MODEL_PATH)
models = artifact["models"]
encoder = artifact["encoder"]
feature_columns = artifact["feature_columns"]
metrics = artifact["metrics"]

st.title("Dry Bean Dataset Classification")
st.write("Compare six classifiers and make predictions using the trained models.")

st.subheader("Evaluation Metrics")
st.dataframe(metrics, use_container_width=True)

st.subheader("Prediction")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    input_df = pd.read_csv(SAMPLE_DATA_PATH) if SAMPLE_DATA_PATH.exists() else pd.DataFrame(columns=feature_columns)

if input_df.empty:
    st.warning("No input data found. Add a CSV file or use the sample test data file.")
    st.stop()

missing_columns = [col for col in feature_columns if col not in input_df.columns]
if missing_columns:
    st.error(f"Missing required columns: {missing_columns}")
    st.stop()

for col in feature_columns:
    input_df[col] = pd.to_numeric(input_df[col], errors="coerce")

model_name = st.selectbox("Choose a model", list(models.keys()))
model = models[model_name]

X = input_df[feature_columns].astype(float)
predicted_ids = model.predict(X)
predicted_labels = encoder.inverse_transform(predicted_ids)

results = input_df.copy()
results["predicted_class"] = predicted_labels

if "Class" in input_df.columns:
    results["actual_class"] = input_df["Class"]

st.dataframe(results.head(20), use_container_width=True)

csv_bytes = results.to_csv(index=False).encode("utf-8")
st.download_button("Download predictions", csv_bytes, file_name="predictions.csv", mime="text/csv")
