# Dry Bean Classification Analysis

## a. Problem Statement
The aim of this project is to classify dry bean varieties using supervised machine learning models and compare their performance using evaluation metrics such as accuracy, AUC, precision, recall, F1 score, and MCC. The goal is to identify which classifier performs best on the Dry Bean dataset and to understand how each model behaves for each bean class.

## b. Dataset Description
The dataset used in this project is the Dry Bean Dataset, which contains multiple physical features of dry beans such as shape, size, and color-related measurements. The target variable is the bean class, which includes several categories such as BARBUNYA, BOMBAY, CALI, DERMASON, HOROZ, SEKER, and SIRA. The dataset was split into training and testing sets to evaluate the models fairly.

## c. GitHub Repository Link
GitHub repository link has not been configured in the current workspace. Once the project is pushed to GitHub, add the repository URL here.

## d. Models Used
The following five classification models were implemented and evaluated:
- Logistic Regression
- Decision Tree Classifier
- k-Nearest Neighbors (kNN)
- Naive Bayes
- Random Forest (Ensemble)

## Comparison Table of Evaluation Metrics
Note: The values below are the weighted metrics reported by the models on the test set.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9214 | 0.9948 | 0.9222 | 0.9214 | 0.9216 | 0.9050 |
| Decision Tree | 0.8920 | 0.9450 | 0.8917 | 0.8920 | 0.8916 | 0.8696 |
| kNN | 0.9166 | 0.9833 | 0.9174 | 0.9166 | 0.9168 | 0.8992 |
| Naive Bayes | 0.7639 | 0.9672 | 0.7654 | 0.7639 | 0.7615 | 0.7154 |
| Random Forest (Ensemble) | 0.9199 | 0.9933 | 0.9199 | 0.9199 | 0.9198 | 0.9032 |

## Class-wise Analysis of Precision, Recall, F1 Score, and Support
Support means the number of actual test samples belonging to that class. For example, if the support for BARBUNYA is 265, it means 265 test samples were truly BARBUNYA.

### 1. Logistic Regression
| Class | Precision | Recall | F1 Score | Support |
|---|---:|---:|---:|---:|
| BARBUNYA | 0.95 | 0.89 | 0.92 | 265 |
| BOMBAY | 1.00 | 1.00 | 1.00 | 104 |
| CALI | 0.93 | 0.94 | 0.94 | 326 |
| DERMASON | 0.92 | 0.91 | 0.92 | 709 |
| HOROZ | 0.96 | 0.95 | 0.96 | 386 |
| SEKER | 0.93 | 0.95 | 0.94 | 406 |
| SIRA | 0.85 | 0.88 | 0.86 | 527 |

Observation: Logistic Regression achieved excellent results across all categories, especially for BOMBAY and HOROZ. It performed slightly weaker on SIRA, which indicates some overlap between SIRA and similar bean classes.

### 2. Decision Tree
| Class | Precision | Recall | F1 Score | Support |
|---|---:|---:|---:|---:|
| BARBUNYA | 0.88 | 0.91 | 0.89 | 265 |
| BOMBAY | 1.00 | 1.00 | 1.00 | 104 |
| CALI | 0.92 | 0.91 | 0.92 | 326 |
| DERMASON | 0.88 | 0.90 | 0.89 | 709 |
| HOROZ | 0.94 | 0.91 | 0.93 | 386 |
| SEKER | 0.91 | 0.95 | 0.93 | 406 |
| SIRA | 0.83 | 0.79 | 0.81 | 527 |

Observation: The Decision Tree performed well but was less stable than the stronger models. It achieved perfect results for BOMBAY, but SIRA had the lowest F1 score, showing that a single tree can struggle with more complex class boundaries.

### 3. k-Nearest Neighbors (kNN)
| Class | Precision | Recall | F1 Score | Support |
|---|---:|---:|---:|---:|
| BARBUNYA | 0.95 | 0.88 | 0.91 | 265 |
| BOMBAY | 1.00 | 1.00 | 1.00 | 104 |
| CALI | 0.92 | 0.94 | 0.93 | 326 |
| DERMASON | 0.91 | 0.91 | 0.91 | 709 |
| HOROZ | 0.95 | 0.95 | 0.95 | 386 |
| SEKER | 0.95 | 0.94 | 0.95 | 406 |
| SIRA | 0.84 | 0.87 | 0.86 | 527 |

Observation: kNN performed very well and gave strong results for most classes. It was also highly accurate for BOMBAY and HOROZ, but SIRA remained a harder class to classify.

### 4. Naive Bayes
| Class | Precision | Recall | F1 Score | Support |
|---|---:|---:|---:|---:|
| BARBUNYA | 0.68 | 0.46 | 0.55 | 265 |
| BOMBAY | 1.00 | 1.00 | 1.00 | 104 |
| CALI | 0.68 | 0.83 | 0.75 | 326 |
| DERMASON | 0.87 | 0.83 | 0.85 | 709 |
| HOROZ | 0.78 | 0.79 | 0.79 | 386 |
| SEKER | 0.68 | 0.71 | 0.70 | 406 |
| SIRA | 0.73 | 0.76 | 0.75 | 527 |

Observation: Naive Bayes had the weakest performance overall. It performed very well for BOMBAY, but it struggled for BARBUNYA and SEKER, which suggests that the feature independence assumption is not ideal for this dataset.

### 5. Random Forest (Ensemble)
| Class | Precision | Recall | F1 Score | Support |
|---|---:|---:|---:|---:|
| BARBUNYA | 0.93 | 0.89 | 0.91 | 265 |
| BOMBAY | 1.00 | 1.00 | 1.00 | 104 |
| CALI | 0.94 | 0.94 | 0.94 | 326 |
| DERMASON | 0.91 | 0.92 | 0.91 | 709 |
| HOROZ | 0.97 | 0.95 | 0.96 | 386 |
| SEKER | 0.94 | 0.96 | 0.95 | 406 |
| SIRA | 0.86 | 0.85 | 0.86 | 527 |

Observation: Random Forest delivered strong and balanced performance across all classes. It performed almost as well as Logistic Regression, showing that ensemble learning is effective for this dataset.

## Observations on Model Performance
| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Best overall performer with the highest accuracy, AUC, and MCC. It produced very strong and balanced results for nearly all bean classes. |
| Decision Tree | Performed reasonably well but was less accurate and less stable than the other models, especially for SIRA. |
| kNN | Showed strong performance and high AUC, but its accuracy was slightly below Logistic Regression. |
| Naive Bayes | The weakest model on this dataset because its assumptions about feature independence did not fit the data well. |
| Random Forest (Ensemble) | Very strong performance and a good alternative to Logistic Regression, with robust results across all classes. |

## Overall Winner for This Dataset
The overall winner for this dataset is Logistic Regression because it achieved the highest accuracy, the highest AUC, and the highest MCC among all tested models. Random Forest was a very close second and also performed exceptionally well.
