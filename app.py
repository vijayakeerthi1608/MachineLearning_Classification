from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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

# Sidebar Theme Selector
with st.sidebar:
    st.subheader("Theme Options")
    theme_mode = st.radio("App Mode", ["Dark Mode", "Light Mode"], index=0)
    st.markdown("---")

is_dark = theme_mode == "Dark Mode"
plotly_template = "plotly_dark" if is_dark else "plotly_white"

# Apply Custom Light/Dark CSS Adjustments
if is_dark:
    st.markdown(
        """
        <style>
            .stApp { background-color: #0e1117; color: #fafafa; }
            .stMetric { background-color: #1e2227; padding: 10px; border-radius: 8px; }
        </style>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """
        <style>
            .stApp { background-color: #f8f9fa; color: #212529; }
            .stMetric { background-color: #ffffff; padding: 10px; border-radius: 8px; border: 1px solid #dee2e6; }
        </style>
        """,
        unsafe_allow_html=True,
    )


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


st.title("🌱 Dry Bean Dataset Classification")
st.write("Upload a test CSV file to evaluate all trained classifiers, compare their results, and inspect metrics.")

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

st.subheader("2. Comparison of All Models")

tab1, tab2 = st.tabs(["📊 Metric Charts", "📋 Data Table"])

with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        # Grouped Bar Chart Comparing All Metrics Across Models
        fig_grouped = px.bar(
            comparison_df.reset_index().melt(id_vars="index", var_name="Metric", value_name="Score"),
            x="index",
            y="Score",
            color="Metric",
            barmode="group",
            title="Overview: Model Comparison Across All Metrics",
            labels={"index": "Model", "Score": "Metric Score"},
            template=plotly_template,
        )
        fig_grouped.update_layout(yaxis=dict(range=[0, 1.05]))
        st.plotly_chart(fig_grouped, use_container_width=True)

    with col2:
        # Horizontal Bar Chart Specifically for Accuracy
        acc_df = comparison_df[["Accuracy"]].sort_values(by="Accuracy", ascending=True).reset_index()
        fig_acc = px.bar(
            acc_df,
            x="Accuracy",
            y="index",
            orientation="h",
            text="Accuracy",
            title="Model Accuracy Comparison",
            labels={"index": "Model", "Accuracy": "Accuracy Score"},
            color="Accuracy",
            color_continuous_scale="Viridis" if is_dark else "Blues",
            template=plotly_template,
        )
        fig_acc.update_layout(xaxis=dict(range=[0, 1.05]))
        st.plotly_chart(fig_acc, use_container_width=True)

with tab2:
    st.dataframe(comparison_df.style.highlight_max(axis=0, color="#1b4332" if is_dark else "#d8f3dc"), use_container_width=True)

# Detailed Inspection
selected_model = st.selectbox("Choose a model to inspect", list(results.keys()))
selected_result = results[selected_model]

st.subheader(f"3. Detailed results for {selected_model}")

# Display KPI cards for selected model
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Accuracy", f"{selected_result['Accuracy']:.4f}")
m2.metric("Precision", f"{selected_result['Precision']:.4f}")
m3.metric("Recall", f"{selected_result['Recall']:.4f}")
m4.metric("F1 Score", f"{selected_result['F1']:.4f}")
m5.metric("MCC", f"{selected_result['MCC']:.4f}")

col_left, col_right = st.columns(2)
with col_left:
    st.write("**Confusion Matrix**")
    # Heatmap visualization for Confusion Matrix
    fig_cm = px.imshow(
        selected_result["cm"],
        x=CLASS_LABELS,
        y=CLASS_LABELS,
        text_auto=True,
        labels=dict(x="Predicted Class", y="Actual Class", color="Count"),
        color_continuous_scale="Purples" if is_dark else "Oranges",
        template=plotly_template,
    )
    fig_cm.update_layout(title_text="Confusion Matrix Heatmap")
    st.plotly_chart(fig_cm, use_container_width=True)

with col_right:
    st.write("**Classification Report**")
    report_df = pd.DataFrame(selected_result["report"]).T
    st.dataframe(report_df.style.format("{:.4f}", na_rep="-"), use_container_width=True)

st.subheader("4. Predicted classes preview")
predicted_labels = [CLASS_LABELS[idx] for idx in selected_result["y_pred"]]
preview_df = dataset.copy()
preview_df["Predicted_Class"] = predicted_labels
preview_df["Actual_Class"] = preview_df[target_column]
st.dataframe(preview_df.head(20), use_container_width=True)