# Build Explainable Risk Prediction for Patient Readmission

## Target Framing

The original "readmitted" column contains three categories: <30 (readmitted within 30 days), >30 (readmitted after 30 days), and NO (not readmitted). Since the goal of the project is specifically to predict early readmission (within 30 days), this variable was converted into a binary target. The <30 category was encoded as 1 (risk), while >30 and NO were merged into 0 (no risk). This transformation turns the problem into a straightforward, interpretable binary classification task and makes it easier to evaluate results.

## Data Handling

The "isnull()" function did not show any missing values, because in the "weight" column (and others), missing entries were encoded as the "?" symbol instead of NaN. Since "?" gives us no usable information, we used "df.isin(['?']).sum()" to check how many hidden missing values exist in each column.

Based on the results:
- The "weight" column was dropped entirely (~97% missing, provided no reliable information)
- The "payer_code" column was dropped (~40% missing, not directly related to clinical condition)
- The "race" column was dropped, since this is a sensitive attribute, and it was removed in advance to prevent the model from producing biased outcomes
- "diag_1", "diag_2", "diag_3" and "medical_specialty" had low missing rates, so these columns were kept, and "?" values were replaced with an "Unknown" category
- "encounter_id" was dropped since it is a simple identifier that contributes nothing to the model

The "discharge_disposition_id" column tells us how a patient left the hospital. We obtained the meaning of these codes from the "IDS_mapping.csv" file. Since our goal is to predict whether a patient returns within 30 days, and a deceased patient cannot physically return, we removed these rows (codes 11, 13, 14, 19, 20 — Expired and Hospice cases). After removal, 99,343 rows remained (down from 101,766).

In the "gender" column, a small number of "Unknown/Invalid" values were found; these rows were dropped.

Since the "age" column contained string values, it would cause issues during model application. Assigning an ordinal midpoint value to each age group is more appropriate, as this preserves a meaningful distance/order between age groups.

## Leakage / Duplicate Patients

To prevent data leakage, we first extracted the unique list of patients. We had previously discovered that a certain number of patients had multiple visits (99,343 rows, but only 69,990 unique patients — meaning 29,353 rows belong to repeat patients). However, since these visits occurred on different dates and for different reasons, they were not deleted.

By performing the train/test split based on the unique patient list, we ensure that all records belonging to the same patient stay on the same side (either fully in train or fully in test) — they are never split between the two. This way, the model is never tested on a patient it has already seen during training, and data leakage is prevented.

After the split operation, we now have two separate dataframes — "train_df" and "test_df". Since the "patient_nbr" column only served the purpose of enabling a leakage-free split, and has already fulfilled that role, it was dropped from both dataframes separately. This column will not be used as a feature in the model, since it is merely an identifier with no clinical meaning.

Since the "readmitted_binary" column already exists, the "readmitted" column was dropped from both dataframes.

## Diagnosis Grouping

The dtypes output showed that most of the remaining columns were of "object" type. Since "diag_1", "diag_2", and "diag_3" contained hundreds of unique ICD-9 diagnosis codes, one-hot encoding created an excessive number of columns (2,420) — a high-cardinality problem. To address this, diagnosis codes were mapped to broad clinical categories (Circulatory, Respiratory, Digestive, Diabetes, Injury, Musculoskeletal, Genitourinary, Neoplasms, Other/Unknown) using the official ICD-9-CM classification code ranges. This reduced the number of columns from 2,420 to 199, and also made the results more clinically meaningful.

When applying one-hot encoding, "train" and "test" were temporarily concatenated ("combined") so that the resulting columns would match between the two sets. "pd.get_dummies()" automatically converted all text (object) columns into numeric form, after which the "is_train" flag was used to split the data back into "train_df" and "test_df".

## Categorical ID Columns — Correction

After the initial model results turned out weaker than expected, the encoding process was reviewed again. It was discovered that "admission_type_id", "discharge_disposition_id", and "admission_source_id" are actually categorical identifiers with no meaningful numerical order between them. However, since these columns were stored as numeric (int64) types in the dataset, "pd.get_dummies()" did not automatically detect and one-hot encode them — they were passed to the model as raw numbers.

This could have caused the model to assume a "greater-than/less-than" relationship between these codes that does not actually exist. To fix this, these three columns were converted to string type using ".astype(str)", and the encoding process was repeated. After the fix, model performance improved only marginally (ROC-AUC: 0.647 → 0.648) — contrary to the initial assumption, this showed that incorrect encoding of these three columns was not the main performance bottleneck.

## Class Imbalance Handling

After binarizing the target variable, the class balance was checked: 0 (no risk) — 70,430, 1 (risk) — 9,024. This is a clear imbalance of roughly 88.6% / 11.4%. If class imbalance is not accounted for, a model could simply predict "0" every time and still show high accuracy, but this would be useless, since the actual goal is to identify at-risk (1) patients.

To address this:
- "class_weight='balanced'" was used for Logistic Regression
- "scale_pos_weight" (70430/9024 ≈ 7.8) was calculated and applied for XGBoost

Balancing was applied only as a model parameter; the underlying data itself was not modified through resampling.

## Baseline Comparison

To account for class imbalance, a Logistic Regression model was trained using "class_weight='balanced'". Test results: recall (class 1) = 0.54, precision (class 1) = 0.17, ROC-AUC = 0.65.

An XGBoost model was then trained on the same data, balanced using "scale_pos_weight". Results: recall (class 1) = 0.51, precision (class 1) = 0.18, ROC-AUC = 0.65.

With default parameters, XGBoost did not outperform the baseline — the results are nearly identical, and even slightly weaker in terms of ROC-AUC and recall. This shows that a more complex model does not automatically translate into better performance, and confirms that gains should be measured rather than assumed.

## Honest Evaluation

Because of the class imbalance, relying on accuracy alone would be misleading (a model predicting "0" every time would still achieve ~88% accuracy). For this reason, the main focus was placed on recall, precision, and overall ROC-AUC for class 1 (risk):

- Recall = 0.51: only about half of the truly at-risk patients are captured
- Precision = 0.18: out of every 100 "at-risk" predictions, only about 18 are correct
- ROC-AUC ≈ 0.65: moderate discriminative ability

These results show that the model is far from perfect, but this reflects the fact that predicting clinical readmission is a genuinely difficult problem, which is also well documented in the literature.

## Model Interpretability (SHAP)

The SHAP global summary plot shows that the model primarily relies on clinically meaningful variables: the number of prior inpatient visits ("number_inpatient") is the strongest predictor — a clinically expected finding (patients who have been hospitalized frequently before are at higher risk of readmission). Other important variables — age, number of medications, number of lab procedures, length of hospital stay, number of diagnoses, and diagnosis categories (Circulatory, Respiratory, Diabetes) — all show clinically sensible relationships.

Local SHAP analysis was performed for three different patients:
- For two patients predicted as low risk (f(x)=-0.48, f(x)=-0.54), the main factor lowering the prediction was "number_inpatient=0" (no prior inpatient visits)
- For the patient predicted as highest risk (f(x)=3.16), the factor most increasing the prediction was "number_inpatient=10" (10 prior inpatient visits), with diagnoses falling under the "Neoplasms" category also contributing to the increased risk

Among the model's most important variables, no suspicious, non-clinical, or potentially biased signal (e.g., ID artifacts, sensitive attributes) was identified.

## Trust Conclusion

This project built an XGBoost-based model to predict the risk of early (within 30 days) hospital readmission for diabetic patients, and used SHAP to explain its decisions.

In terms of performance, XGBoost (ROC-AUC≈0.65) did not meaningfully outperform the baseline Logistic Regression model (ROC-AUC≈0.65). This demonstrated that a more complex model does not automatically provide an advantage, and revealed the limited predictive signal available in the current feature set.

In terms of trustworthiness, however, the SHAP analysis showed that the model relies primarily on clinically meaningful variables (prior inpatient visit count, age, medication count, diagnosis severity). No suspicious or non-clinical signal was found among the model's top drivers. This suggests the model rests on trustworthy foundations, even though its overall predictive power (ROC-AUC≈0.65) is not high and leaves room for future improvement (hyperparameter tuning, additional feature engineering, and properly encoding remaining categorical ID columns such as "admission_type_id").

## Limitations / Future Work

- The encoding fix for columns like "admission_type_id" and "admission_source_id" was applied late and only marginally improved performance — this suggests the main bottleneck is the limited predictive power of the available features, rather than data complexity alone
- XGBoost hyperparameters (max_depth, learning_rate, n_estimators) were not tuned and were left at default values — tuning could likely improve results
- Since the sensitive attribute "race" was dropped early in preprocessing, a subgroup fairness analysis was not performed in this version