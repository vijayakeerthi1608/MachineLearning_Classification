from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
SAMPLE_DATA_PATH = ROOT / "test_data.csv"

DEFAULT_FEATURE_COLUMNS = [
    "Area",
    "Perimeter",
    "MajorAxisLength",
    "MinorAxisLength",
    "AspectRation",
    "Eccentricity",
    "ConvexArea",
    "EquivDiameter",
    "Extent",
    "Solidity",
    "roundness",
    "Compactness",
    "ShapeFactor1",
    "ShapeFactor2",
    "ShapeFactor3",
    "ShapeFactor4",
]
CLASS_LABELS = ["BARBUNYA", "BOMBAY", "CALI", "DERMASON", "HOROZ", "SEKER", "SIRA"]

st.set_page_config(page_title="Dry Bean Classifier", page_icon="🌱", layout="wide")


@st.cache_data(show_spinner=False)
def load_models() -> dict:
    model_files = {
        "Logistic Regression": MODEL_DIR / "logistic_regression.joblib",
        "Decision Tree": MODEL_DIR / "decision_tree.joblib",
        "kNN": MODEL_DIR / "knn_classifier.joblib",
        "Naive Bayes": MODEL_DIR / "naive_bayes.joblib",
        "Random Forest": MODEL_DIR / "random_forest.joblib",
    }

    missing = [name for name, path in model_files.items() if not path.exists()]
    if missing:
        st.error(f"Missing model files: {', '.join(missing)}")
        st.stop()

    return {name: joblib.load(path) for name, path in model_files.items()}


@st.cache_data(show_spinner=False)
def load_dataset(uploaded_file) -> tuple[pd.DataFrame, list[str], str]:
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_csv(SAMPLE_DATA_PATH) if SAMPLE_DATA_PATH.exists() else pd.DataFrame()

    if df.empty:
        raise ValueError("Please upload a CSV file containing test data with the required columns.")

    available_features = [col for col in DEFAULT_FEATURE_COLUMNS if col in df.columns]
    if not available_features:
        raise ValueError("The uploaded dataset does not contain the required feature columns.")

    target_column = "Class" if "Class" in df.columns else None
    if target_column is None:
        raise ValueError("The uploaded dataset must include a Class column for evaluation.")

    prepared_df = df[available_features + [target_column]].copy()
    for col in available_features:
        prepared_df[col] = pd.to_numeric(prepared_df[col], errors="coerce")

    prepared_df = prepared_df.dropna(subset=available_features + [target_column]).reset_index(drop=True)
    if prepared_df.empty:
        raise ValueError("No usable rows remain after cleaning the uploaded data.")

    return prepared_df, available_features, target_column


@st.cache_data(show_spinner=False)
def evaluate_models(dataset: pd.DataFrame, feature_columns: list[str], target_column: str) -> tuple[pd.DataFrame, dict]:
    label_encoder = LabelEncoder()
    label_encoder.fit(CLASS_LABELS)
    y_true = label_encoder.transform(dataset[target_column].astype(str))

    X = dataset[feature_columns].astype(float)
    results = {}

    for model_name, model in load_models().items():
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X) if hasattr(model, "predict_proba") else None

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        mcc = matthews_corrcoef(y_true, y_pred)
        auc = roc_auc_score(y_true, y_prob, multi_class="ovr") if y_prob is not None else None

        results[model_name] = {
            "Accuracy": accuracy,
            "AUC": auc if auc is not None else float("nan"),
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "MCC": mcc,
            "y_pred": y_pred,
            "y_prob": y_prob,
            "cm": confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_LABELS)))),
            "report": classification_report(
                y_true,
                y_pred,
                labels=list(range(len(CLASS_LABELS))),
                target_names=CLASS_LABELS,
                output_dict=True,
                zero_division=0,
            ),
        }

    comparison_df = pd.DataFrame(
        {
            model_name: {
                "Accuracy": values["Accuracy"],
                "AUC": values["AUC"],
                "Precision": values["Precision"],
                "Recall": values["Recall"],
                "F1": values["F1"],
                "MCC": values["MCC"],
            }
            for model_name, values in results.items()
        }
    ).T.round(4)

    return comparison_df, results


st.title("Dry Bean Dataset Classification")
st.write("Upload a test CSV file to evaluate all trained classifiers and compare their results.")

with st.sidebar:
    st.subheader("Upload test data")
    uploaded_file = st.file_uploader("Upload a CSV file (test data only)", type=["csv"])
    st.caption("Use a CSV file that contains the bean features and a Class column.")

try:
    dataset, feature_columns, target_column = load_dataset(uploaded_file)
except ValueError as exc:
    st.warning(str(exc))
    st.stop()

st.subheader("1. Dataset preview")
st.dataframe(dataset.head(), use_container_width=True)

comparison_df, results = evaluate_models(dataset, feature_columns, target_column)

st.subheader("2. Comparison of all models")
st.dataframe(comparison_df, use_container_width=True)

selected_model = st.selectbox("Choose a model to inspect", list(results.keys()))
selected_result = results[selected_model]

st.subheader(f"3. Detailed results for {selected_model}")

col1, col2 = st.columns(2)
with col1:
    st.write("Confusion Matrix")
    cm_df = pd.DataFrame(
        selected_result["cm"],
        index=CLASS_LABELS,
        columns=CLASS_LABELS,
    )
    st.dataframe(cm_df, use_container_width=True)

with col2:
    st.write("Classification Report")
    report_df = pd.DataFrame(selected_result["report"]).T
    st.dataframe(report_df, use_container_width=True)

st.subheader("4. Predicted classes preview")
predicted_labels = [CLASS_LABELS[idx] for idx in selected_result["y_pred"]]
preview_df = dataset.copy()
preview_df["Predicted_Class"] = predicted_labels
preview_df["Actual_Class"] = preview_df[target_column]
st.dataframe(preview_df.head(20), use_container_width=True)
