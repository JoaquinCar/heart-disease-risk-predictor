# Heart Disease Risk Predictor

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3-orange)
![SHAP](https://img.shields.io/badge/SHAP-0.44-red)

## Overview

An end-to-end machine learning project for predicting coronary 
artery disease (CAD) using the UCI Heart Disease dataset 
(920 patients, 4 institutions). Built as a first step into 
Clinical AI research, with emphasis on clinical reasoning, 
documented decision-making, and model interpretability.

Every analytical decision — from missing value treatment to 
feature exclusion — is grounded in clinical context rather 
than treated as abstract data processing.

**Stack:** Python, Pandas, NumPy, Scikit-learn, Matplotlib, 
Seaborn, SHAP, missingno

---

## Dataset

**Source:** UCI Heart Disease Dataset — combined version  
- UCI Repository: https://archive.ics.uci.edu/dataset/45/heart+disease  
- Kaggle: https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data

| | |
|---|---|
| Total patients | 920 |
| Institutions | 4 (Cleveland, Hungary, Switzerland, VA Long Beach) |
| Original paper | Detrano et al., *American Journal of Cardiology*, 1989 |
| Features | 16 original → 11 used for modeling |
| Missing values | 0 (after iterative imputation) |
| Target variable | `target` — binary (0 = no CAD, 1 = CAD present) |

---

## Methodology

### 1. Exploratory Data Analysis
The target variable `num` (0–4 severity scale) was binarized to 
0/1 — the clinical threshold for significant CAD is ≥50% arterial 
stenosis, an inherently binary decision. This yielded a near-balanced 
distribution: 411 negative (44.7%) vs 509 positive (55.3%).

Key findings per variable type:

**Continuous variables:** `thalch` and `oldpeak` showed the 
strongest separation between CAD and no-CAD groups — both 
derived from exercise stress testing and directly measuring 
cardiac response under exertion. `age` showed moderate 
discriminative power with a clear trend after 45 years.

**Categorical variables:** `cp`, `exang`, and `slope` were 
the strongest predictors. Notably, asymptomatic chest pain 
showed 79% CAD prevalence — a counterintuitive finding 
explained by referral bias (clinicians had high suspicion 
before ordering angiography). `slope` had 309 missing values 
but was retained due to strong predictive power.

**Correlation analysis:** Spearman matrix confirmed `cp` 
(-0.47), `exang` (0.46), `thalch` (-0.40), and `oldpeak` 
(0.40) as the strongest correlates with target.

### 2. Preprocessing
- Variables with >50% missingness (`ca`, `thal`) excluded
- `dataset` column excluded to prevent data leakage
- Impossible values recoded as NaN (`chol=0`, `trestbps=0`)
- Categorical encoding via OrdinalEncoder
- Missing values imputed using IterativeImputer (max_iter=10)
- StandardScaler applied for Logistic Regression only

### 3. Modeling
Four models evaluated using Stratified K-Fold (k=5) with 
20% hold-out set reserved for final evaluation:

| Model | ROC-AUC (CV) |
|-------|-------------|
| Logistic Regression | 0.868 |
| Decision Tree | 0.737 |
| **Random Forest (tuned)** | **0.895** |
| XGBoost (tuned) | 0.885 |

**Random Forest** selected as best model with hyperparameters: 
`max_depth=5`, `min_samples_split=10`, `n_estimators=200`.

### 4. Final Evaluation (Hold-out Set)

| Metric | Score |
|--------|-------|
| ROC-AUC | 0.920 |
| F1 | 0.859 |
| Precision | 0.854 |
| Recall | 0.863 |
| False Negatives | 14 / 102 CAD patients |

### 5. Interpretability (SHAP)
SHAP analysis revealed `slope`, `cp`, and `exang` as the 
top 3 predictors — all related to exercise stress testing. 
`slope` ranked 1st in SHAP despite ranking 5th in the EDA, 
explained by the iterative imputation enabling the model to 
learn from all 920 complete observations.

### 6. Fairness Evaluation

| Metric | Male (n=149) | Female (n=35) |
|--------|-------------|---------------|
| ROC-AUC | 0.911 | 0.917 |
| F1 | 0.878 | 0.625 |
| Recall | 0.883 | 0.625 |

Both groups achieve ROC-AUC >0.91. However, F1 and Recall 
drop substantially for female patients due to insufficient 
training data (194 females, 21% of dataset) and atypical 
CAD presentation patterns in women underrepresented in 
this dataset.

---

## Key Findings

1. **Stress test variables dominate:** The 5 strongest predictors 
   (`slope`, `cp`, `exang`, `oldpeak`, `thalch`) all relate to 
   physiological response under exercise — more informative than 
   resting measurements for CAD prediction.

2. **Counterintuitive clinical pattern:** Asymptomatic patients 
   showed the highest CAD prevalence (79%) — explained by referral 
   bias in a population already selected for high clinical suspicion.

3. **Institutional heterogeneity:** CAD prevalence varied from 
   45.7% (Cleveland) to 93.5% (Switzerland) across institutions, 
   confirming the `dataset` column must be excluded to avoid 
   data leakage.

4. **Imputation enabled stronger modeling:** `slope` became the 
   top SHAP predictor after iterative imputation filled 309 missing 
   values — demonstrating that rigorous missing data handling 
   directly improves model performance.

---

## Limitations

1. **Referral bias:** All patients were referred for coronary 
   angiography — CAD prevalence (55.3%) far exceeds general 
   population rates (~6-7%). Not suitable for general screening 
   without recalibration.

2. **Female underrepresentation:** Only 194 female patients (21%). 
   Model performance for women should be interpreted cautiously 
   and validated on larger sex-balanced datasets.

3. **Missing stress test data:** The strongest predictors require 
   exercise stress testing — not always available in clinical 
   practice. Model performance may degrade without these features.

4. **Cholesterol limitation:** Only total cholesterol available. 
   The LDL/HDL ratio would be clinically more informative.

5. **Institutional generalization:** The model was trained on 
   heterogeneous data from 4 institutions with different protocols 
   and prevalence rates. Performance on a single institution's 
   patient population may vary.

---

## References

1. Detrano, R., et al. (1989). International application of a new 
   probability algorithm for the diagnosis of coronary artery disease. 
   *The American Journal of Cardiology, 64*(5), 304–310.

2. UCI Machine Learning Repository. Heart Disease Dataset. 
   https://archive.ics.uci.edu/dataset/45/heart+disease

3. Whelton, P. K., et al. (2018). 2017 ACC/AHA Hypertension Guideline. 
   *JACC, 71*(19), e127–e248.

4. Grundy, S. M., et al. (2019). 2018 AHA/ACC Cholesterol Guideline. 
   *Circulation, 139*(25), e1082–e1143.

5. American College of Sports Medicine. (2022). 
   *ACSM's Guidelines for Exercise Testing and Prescription* (11th ed.).

6. Rubin, D. B. (1976). Inference and missing data. 
   *Biometrika, 63*(3), 581–592.

7. van Buuren, S. (2018). *Flexible Imputation of Missing Data* (2nd ed.). 
   https://stefvanbuuren.name/fimd/

8. Sterne, J. A. C., et al. (2009). Multiple imputation for missing 
   data in epidemiological and clinical research. *BMJ, 338*, b2393.

9. Collins, G. S., et al. (2024). TRIPOD+AI statement. 
   *BMJ, 385*, e078378.

10. Lundberg, S. M., & Lee, S. I. (2017). A unified approach to 
    interpreting model predictions. *NeurIPS, 30*, 4765–4774.