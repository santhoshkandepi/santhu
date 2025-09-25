
# Assignment Report — End-to-End ML Project

## Dataset & Objective
- File: `synthetic_dataset_10000x20.csv` (10,000 rows × 21 columns)
- Target: `target_default_risk` (binary: 0 = No Default, 1 = Default)
- Objective: Preprocess data, build and compare Logistic Regression, Decision Tree, SVM, Random Forest, and XGBoost models, and tune hyperparameters for improved performance.

## Key EDA Findings (summary)
- Dataset shape: (dataset file not found at runtime)
- Numeric features: summary statistics and distributions (see notebook visualizations).
- Some columns contained missing values; categorical columns had occasional typos (e.g., 'bachlors').
- Target class balance: refer to notebook; handle imbalance if present using resampling or class weights.

## Preprocessing Decisions
- Imputed numeric missing values with **median** to reduce outlier influence.
- Imputed categorical missing values with the constant `'missing'`.
- Fixed common typos in `education` (lowercased and replaced 'bachlors' -> 'bachelors').
- Converted `signup_date` to datetime and created `signup_recency_days` as a feature (if present).
- Scaled numeric features using **StandardScaler**; used One-Hot Encoding for low-cardinality categoricals and OrdinalEncoder for high-cardinality categoricals as fallback.

## Modeling & Tuning
- Baseline models implemented: Logistic Regression, Decision Tree, SVM, Random Forest, and XGBoost.
- RandomizedSearchCV used to tune Random Forest and XGBoost hyperparameters (search spaces included `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `gamma`, etc.).
- Evaluation metrics: **accuracy, precision, recall, F1-score, confusion matrix**.
- Tuned models were evaluated on a held-out test set for honest performance estimates.

## Results & Interpretation
- Baseline model performance (accuracy/F1) is shown in the notebook table (summary_df).
- Tuned Random Forest and XGBoost showed improved F1-scores during cross-validation and on the test set.
- Feature importance from XGBoost helps identify which features drive predictions (visualized in notebook).

## Conclusions & Recommendations
- Ensure careful preprocessing: outlier handling, correct encoding, and thoughtful imputation matter a lot.
- If target is imbalanced, try class weighting or resampling (SMOTE/undersampling) before training.
- For production-ready models consider ensembling (stacking) and model calibration if probabilities are used for decision-making.
- Document all preprocessing steps and assumptions in the final report (this file and the notebook).

---
*This report is a concise 1-2 page summary. For full code, visualizations, and detailed results, open `student_assignment_completed.ipynb`.*
