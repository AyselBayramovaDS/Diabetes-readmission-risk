# Explainable Risk Prediction for Patient Readmission

Predicting early (within 30 days) hospital readmission risk for diabetic patients using XGBoost, with model decisions explained through SHAP.

## Overview

This project uses the [Diabetes 130-US Hospitals dataset](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008) (1999–2008, ~101,766 inpatient encounters) to build a binary classification model that predicts whether a diabetic patient will be readmitted within 30 days of discharge. The project focuses not just on predictive performance, but on trustworthy data handling and model interpretability.

## Project Structure

```
.
├── riskprediction.ipynb   # Main notebook: EDA, preprocessing, modeling, SHAP
├── note.md                # Documentation of decisions and reasoning
├── diabetic_data.csv      # Raw dataset (not included — see Data section)
├── IDS_mapping.csv        # ID-to-description mapping for categorical codes
└── README.md
```

## Data

The dataset is not included in this repository due to size/licensing. Download it from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008) and place `diabetic_data.csv` and `IDS_mapping.csv` in the project root before running the notebook.

## Approach

1. **Target framing** — the original 3-category `readmitted` column (`<30`, `>30`, `NO`) is binarized into early readmission (1) vs. not (0)
2. **Data cleaning** — hidden missing values (`"?"`), high-missing columns, and sensitive attributes are handled explicitly
3. **Leakage prevention** — records of deceased/hospice patients are removed, and train/test split is performed at the patient level (not row level) to avoid the same patient appearing in both sets
4. **Diagnosis grouping** — high-cardinality ICD-9 diagnosis codes are mapped to broad clinical categories
5. **Class imbalance handling** — addressed via `class_weight`/`scale_pos_weight` rather than resampling
6. **Baseline comparison** — Logistic Regression baseline vs. XGBoost, evaluated with the same metrics
7. **Evaluation** — precision, recall, F1, and ROC-AUC (accuracy is misleading given the class imbalance)
8. **Interpretability** — SHAP global summary plot and local (per-patient) explanations

See [`note.md`](./note.md) for the full reasoning behind each decision, including data quality issues found, class imbalance handling, baseline comparison, and the trust conclusion drawn from SHAP.

## Requirements

```
pandas
numpy
scikit-learn
xgboost
shap
matplotlib
```

Install with:
```bash
pip install pandas numpy scikit-learn xgboost shap matplotlib
```

## Running

Open `riskprediction.ipynb` in Jupyter or VS Code and run all cells in order.

## Key Results

- **Baseline (Logistic Regression):** ROC-AUC ≈ 0.65, Recall (class 1) ≈ 0.54
- **XGBoost:** ROC-AUC ≈ 0.65, Recall (class 1) ≈ 0.51
- XGBoost did not meaningfully outperform the baseline with default parameters
- SHAP analysis shows the model relies primarily on clinically meaningful signals (prior inpatient visits, age, medication count, diagnosis severity), with no suspicious or non-clinical signal among the top drivers

## Limitations

- XGBoost hyperparameters were not tuned
- Subgroup fairness analysis was not performed (the `race` attribute was dropped early in preprocessing)

See `note.md` for full details.
