# 🌱 Dry Bean Dataset Classification Project

An end-to-end Machine Learning web application built with **Streamlit**, **Scikit-Learn**, and **Plotly** to classify dry bean varieties based on geometric features. The app provides interactive model evaluation, class-wise metrics analysis, and dynamic visual comparisons across multiple classification models.

---

## 📋 Table of Contents
- [Project Overview](#-project-overview)
- [Key Features](#-key-features)
- [Models Implemented](#-models-implemented)
- [Repository Structure](#-repository-structure)
- [Local Installation & Setup](#-local-installation--setup)
- [Running the App](#-running-the-app)
- [Deploying to Streamlit Cloud](#-deploying-to-streamlit-cloud)
- [Model Evaluation & Analysis](#-model-evaluation--analysis)
- [Summary of Model Behaviors](#-summary-of-model-behaviors)
- [Overall Winner](#-overall-winner)

---

## 📌 Project Overview
The objective of this project is to classify dry bean seeds into seven distinct varieties (`BARBUNYA`, `BOMBAY`, `CALI`, `DERMASON`, `HOROZ`, `SEKER`, `SIRA`) using physical shape, size, and boundary measurements. 

Five distinct supervised machine learning models are trained and evaluated using standard classification metrics: **Accuracy, AUC, Precision, Recall, F1 Score, and MCC (Matthews Correlation Coefficient)**.

---

## ✨ Key Features
- **Interactive Light / Dark Mode**: Custom theme toggle in the sidebar that dynamically adjusts app styling and Plotly chart color schemes.
- **Visual Model Comparisons**: Interactive grouped and horizontal bar charts comparing model performance metrics side-by-side.
- **Confusion Matrix Heatmaps**: Color-coded, interactive confusion matrices for deep-dive error analysis per model.
- **Custom CSV Predictions**: Upload any test dataset containing the required bean features and target classes to instantly evaluate all trained models.
- **Detailed Class-level Metrics**: Inspect per-class precision, recall, F1-scores, and support counts.

---

## 🤖 Models Implemented
1. **Logistic Regression**
2. **Decision Tree Classifier**
3. **k-Nearest Neighbors (kNN)**
4. **Naive Bayes**
5. **Random Forest Classifier (Ensemble)**

---

## 📁 Repository Structure
```text
.
├── app.py                  # Main Streamlit web application script
├── test_data.csv           # Sample test dataset for evaluations
├── requirements.txt        # Python package dependencies
├── .gitignore              # Ignored files (venv, cache, local configs)
└── model/
    ├── train_models.py     # Script to train and save models
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn_classifier.joblib
    ├── naive_bayes.joblib
    └── random_forest.joblib

```

---

## ⚙️ Local Installation & Setup

### Prerequisites

* Python 3.10+ installed on your system.
* Git installed.

### 1. Clone the Repository

```bash
git clone [https://github.com/vijayakeerthi1608/MachineLearning_Classification.git](https://github.com/vijayakeerthi1608/MachineLearning_Classification.git)
cd MachineLearning_Classification

```

### 2. Create and Activate a Virtual Environment

* **On Windows (PowerShell):**

```powershell
python -m venv venv
.\venv\Scripts\activate

```

* **On macOS / Linux:**

```bash
python3 -m venv venv
source venv/bin/activate

```

### 3. Install Dependencies

```bash
pip install -r requirements.txt

```

---

## 🚀 Running the App

### Option A: Retrain Models (Optional)

If you wish to train the models from scratch and regenerate the `.joblib` files in the `model/` folder:


### Option B: Launch the Streamlit Web App

Run Streamlit via your active Python environment:

```bash
python -m streamlit run app.py

```

Once executed, open your browser and navigate to **`http://localhost:8501`**.

---

## ☁️ Deploying to Streamlit Cloud

This application is fully optimized for **Streamlit Community Cloud**:

1. Push your code to GitHub (ensure `requirements.txt` includes `plotly`, `streamlit`, `pandas`, `scikit-learn`, and `joblib`).
2. Log in to [Streamlit Community Cloud](https://streamlit.io/cloud).
3. Click **New app**, select your repository (`MachineLearning_Classification`), branch (`main`), and set the main file path to `app.py`.
4. Click **Deploy**. Streamlit Cloud will automatically install dependencies from `requirements.txt` and host your live application 24/7!

---

## 📊 Model Evaluation & Analysis

### Overall Performance Comparison (Test Set)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
| --- | --- | --- | --- | --- | --- | --- |
| **Logistic Regression** | **0.9214** | **0.9948** | **0.9222** | **0.9214** | **0.9216** | **0.9050** |
| **Decision Tree** | 0.8920 | 0.9450 | 0.8917 | 0.8920 | 0.8916 | 0.8696 |
| **kNN** | 0.9166 | 0.9833 | 0.9174 | 0.9166 | 0.9168 | 0.8992 |
| **Naive Bayes** | 0.7639 | 0.9672 | 0.7654 | 0.7639 | 0.7615 | 0.7154 |
| **Random Forest** | 0.9199 | 0.9933 | 0.9199 | 0.9199 | 0.9198 | 0.9032 |

---

## 💡 Summary of Model Behaviors

* **Logistic Regression**: **Best overall performer.** Achieved the highest accuracy (92.14%), AUC (0.9948), and MCC (0.9050). Displays robust separation across almost all bean classes, especially `BOMBAY` (100% precision/recall) and `HOROZ`.
* **Random Forest**: Very close second (91.99% accuracy). Provides strong, balanced class-wise results and generalizes extremely well without overfitting.
* **k-Nearest Neighbors (kNN)**: High overall AUC (0.9833) and strong baseline performance (91.66% accuracy), though slightly sensitive to feature scaling in boundary classes like `SIRA`.
* **Decision Tree**: Decent performance (89.20%), but exhibits higher variance and lower stability on complex class intersections.
* **Naive Bayes**: Weakest performer (76.39%). Strong correlations among geometric bean dimensions violate Naive Bayes' fundamental feature-independence assumption.

---

## 🏆 Overall Winner

**Logistic Regression** is the top-performing model for this dataset. It achieves the highest accuracy, AUC, and MCC while maintaining low computational overhead. **Random Forest** serves as an excellent ensemble alternative.
