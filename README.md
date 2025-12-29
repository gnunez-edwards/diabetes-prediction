# Diabetes Prediction Using Machine Learning

## Problem
Diabetes is a widespread chronic condition, and early risk identification can support
preventive care. The goal of this project was to build and evaluate classification models
that predict whether an individual is diabetic (binary outcome) using clinical and
behavioral risk factors.

---

## Data
This project uses the UCI dataset (loaded directly from UCI as a CSV) containing
patient-level features such as:

- **Clinical indicators:** HighBP, HighChol, BMI, Stroke, HeartDiseaseorAttack
- **Behavioral factors:** Smoker, PhysActivity, Fruits, Veggies, HvyAlcoholConsump
- **Access & socioeconomic:** AnyHealthcare, NoDocbcCost, Education, Income
- **Self-reported health:** GenHlth, MentHlth, PhysHlth
- **Demographics:** Sex, Age (category)

Target variable: **Diabetes_binary** (0 = no diabetes, 1 = diabetes)

---

## Approach
This project follows an end-to-end classification workflow:

### 1) Preprocessing
- Train/test split with **stratification**
- Feature scaling using **MinMaxScaler**
- Identified **class imbalance** in the target distribution

### 2) Handling Class Imbalance
To reduce false negatives on the minority class, models were trained and compared under:
- **Baseline training** (scaled data)
- **SMOTE oversampling** (minority class)

### 3) Models Trained
- Logistic Regression
- Decision Tree (including feature selection + hyperparameter tuning with GridSearchCV)
- Naive Bayes (GaussianNB + var_smoothing tuning)
- K-Nearest Neighbors (k tuning via GridSearchCV)
- Ensemble classifier (VotingClassifier, **soft** and **hard** voting)

### 4) Evaluation
Performance was evaluated using:
- Accuracy (overall)
- Precision / Recall / F1 for the positive class (diabetes)
- Confusion matrices
- ROC curves and AUC

---

## Results & Insights
- With imbalanced data, models often achieved high **accuracy** while producing low **recall**
  for the diabetes class (false negatives).
- Applying **SMOTE** generally improved recall for the diabetes class, often at the cost of
  lower overall accuracy.
- Model performance varied by objective:
  - If minimizing missed diabetes cases matters most, **recall**-optimized models performed better under SMOTE.
  - If overall accuracy is prioritized, baseline training performed better but risked under-detecting positives.

A results table comparing models under both settings is included in the notebook.

---

## Tools & Technologies
- Python
- pandas, NumPy
- scikit-learn
- imbalanced-learn (SMOTE)
- matplotlib (visualizations)

---

## Repository Contents
- Notebook / script with:
  - EDA and distribution checks
  - Preprocessing + scaling
  - Baseline vs SMOTE training runs
  - Model tuning + ensemble evaluation
  - Results tables and plots (confusion matrix, ROC curves)

---

## Key Takeaways
This project demonstrates applied machine learning skills including:
class imbalance handling (SMOTE), comparative model evaluation, hyperparameter tuning,
and performance tradeoffs between overall accuracy and minority-class recall.
