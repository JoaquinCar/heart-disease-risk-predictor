# CLAUDE.md — Heart Disease Risk Predictor + MLOps

## Who is Joaquín

Joaquín Carmona (nickname: kino) is a TSU Software Development student at ITS Mérida, graduating September 2026. Currently a DevOps Intern at AIVARA. His long-term goal is a PhD in Healthcare AI (target programs: UW, JHU, Georgia Tech, Michigan, UIUC), with a Canadian MSc thesis as the strategic bridge (Ottawa first choice).

**Learning style:** Attempts code himself first, then asks for guidance. Wants line-by-line understanding before running. Asks conceptual "why" questions frequently. Prefers depth over speed. Informal and conversational. Direct and pragmatic.

**Current context:** Transitioning from DevOps → MLOps while building a Healthcare AI portfolio for PhD applications.

---

## What we built (Project 1 — completed)

A full end-to-end machine learning project for predicting coronary artery disease (CAD) using the UCI Heart Disease dataset.

### Repository structure (current)

```
heart-risk-predictor/
├── 01_EDA.ipynb                 # Full exploratory data analysis
├── 02_modeling.ipynb            # Modeling, SHAP, fairness evaluation
├── heart_disease_processed.csv  # Cleaned dataset (920 rows, 0 missing)
└── README.md                    # Full project documentation
```

### Dataset

- UCI Heart Disease — combined version (Cleveland, Hungary, Switzerland, VA Long Beach)
- 920 patients, 16 original features → 11 used for modeling
- Target: binary CAD diagnosis (0 = no disease, 1 = disease present)
- Source: [https://archive.ics.uci.edu/dataset/45/heart+disease](https://archive.ics.uci.edu/dataset/45/heart+disease)

### Key EDA decisions (all clinically justified)

- `num` binarized to `target` (0/1) — clinical threshold ≥50% stenosis
- `ca` (66.4% missing) and `thal` (52.8% missing) excluded from model
- `dataset` column excluded — institution of origin causes data leakage
- `chol=0` (172 cases) recoded as NaN — physiologically impossible
- `trestbps=0` and `oldpeak<-2` recoded as NaN — implausible values
- Missingness pattern confirmed as MAR (Missing At Random) conditioned on institution
- Iterative imputation (IterativeImputer, max_iter=10) over median imputation

### Features used for modeling (11 total)


| Rank | Feature  | Type        | EDA finding                         |
| ---- | -------- | ----------- | ----------------------------------- |
| 1    | exang    | Categorical | 83.7% CAD when True                 |
| 2    | thalch   | Continuous  | Clear left shift in CAD patients    |
| 3    | cp       | Categorical | Asymptomatic = 79% CAD              |
| 4    | oldpeak  | Continuous  | Direct ischemia measurement         |
| 5    | slope    | Categorical | ~77% CAD in flat/downsloping        |
| 6    | age      | Continuous  | Strong trend after 45               |
| 7    | sex      | Categorical | 63% male vs 26% female CAD rate     |
| 8    | restecg  | Categorical | Moderate predictor                  |
| 9    | trestbps | Continuous  | Weak — heavy overlap                |
| 10   | chol     | Continuous  | Weak — total only, no LDL/HDL       |
| 11   | fbs      | Categorical | Weak — binary encoding loses nuance |


### Preprocessing pipeline

1. OrdinalEncoder for categorical columns (sex, cp, restecg, slope)
2. Boolean to float for fbs, exang
3. IterativeImputer(max_iter=10, random_state=42)
4. StandardScaler — applied only to Logistic Regression (tree models don't need it)
5. train_test_split(test_size=0.2, stratify=y, random_state=42)

### Modeling results


| Model               | ROC-AUC (CV) | ROC-AUC (tuned) |
| ------------------- | ------------ | --------------- |
| Logistic Regression | 0.868        | —               |
| Decision Tree       | 0.737        | —               |
| Random Forest       | 0.882        | **0.895**       |
| XGBoost             | 0.868        | 0.885           |


**Winner: Random Forest**
Best params: max_depth=5, min_samples_split=10, n_estimators=200

### Final hold-out evaluation (184 patients)


| Metric          | Score                 |
| --------------- | --------------------- |
| ROC-AUC         | 0.920                 |
| F1              | 0.859                 |
| Precision       | 0.854                 |
| Recall          | 0.863                 |
| False Negatives | 14 / 102 CAD patients |


### SHAP results

Top features by mean |SHAP value|:

1. slope (0.12) — surpassed EDA rank due to imputation enabling full 920 rows
2. cp (0.10)
3. exang (0.075)
4. oldpeak (0.05)
5. thalch (0.038)

All top 5 are exercise stress test related — confirms that physiological response under stress is more informative than resting measurements.

### Fairness evaluation


| Metric  | Male (n=149) | Female (n=35) |
| ------- | ------------ | ------------- |
| ROC-AUC | 0.911        | 0.917         |
| F1      | 0.878        | 0.625         |
| Recall  | 0.883        | 0.625         |


Gap explained by: insufficient female data (194 total, 21%) and atypical CAD presentation in women underrepresented in training data.

### Visualization conventions (maintain throughout)

- Blue (#2196F3) = No Disease
- Red (#F44336) = Disease
- Institution colors: Yellow=Cleveland, Purple=Hungary, Blue=VA Long Beach, Green=Switzerland

---

## What we are building now (MLOps extension)

The goal is dual:

1. **Short term** — transition DevOps → MLOps at AIVARA, build relevant portfolio
2. **Long term** — PhD Healthcare AI, where MLOps is useful background

### Target repository structure (final)

```
heart-risk-predictor/
├── notebooks/
│   ├── 01_EDA.ipynb
│   └── 02_modeling.ipynb
├── src/
│   ├── train.py              # Training script with MLflow tracking
│   └── predict.py            # Prediction logic
├── api/
│   └── main.py               # FastAPI endpoint
├── tests/
│   └── test_api.py           # Basic API tests
├── .github/
│   └── workflows/
│       └── ci.yml            # GitHub Actions CI pipeline
├── Dockerfile                # Containerize the API
├── requirements.txt
├── heart_disease_processed.csv
└── README.md
```

### MLOps stack (in order of implementation)

**Level 1 — MLflow (experiment tracking)**

- Track every training run: metrics, parameters, model artifacts
- Compare runs visually in MLflow UI
- Save and load the best model from MLflow registry
- Why: standard in both research and industry, directly relevant to AIVARA

**Level 2 — FastAPI (model serving)**

- POST endpoint that receives 11 patient features as JSON
- Returns CAD probability and binary prediction
- Input validation with Pydantic
- Why: demonstrates model can be used beyond a notebook

**Level 3 — Docker (containerization)**

- Dockerfile that packages FastAPI + model into a container
- docker-compose for local development
- Why: connects directly to Joaquín's DevOps background at AIVARA

**Level 4 — GitHub Actions (CI/CD)**

- Pipeline triggered on push to main
- Runs tests automatically
- Builds Docker image
- Why: Joaquín already knows CI/CD from DevOps, this is the natural bridge

**Level 5 — Evidently (data drift monitoring)**

- Basic drift detection comparing new patient data to training distribution
- Report generation
- Why: clinically relevant — patient populations change over time

### Implementation approach

- Build with Claude Code for natural MLOps-like workflow
- Each level is independent and buildable incrementally
- Joaquín and you code together, do not rush, its a learning path, understanding everything, answer his questions always with and educative and learning approach.
- Every decision explained conceptually before implementation
- Clinical context maintained throughout (this is still a healthcare AI project)

---

## Technical environment

- OS: Windows, Git Bash
- Python: virtual environment (source venv/Scripts/activate for Git Bash)
- Hardware: RTX 3050 (4GB VRAM)
- Git: GitHub at [https://github.com/JoaquinCar](https://github.com/JoaquinCar)

## Key principles to maintain

1. Clinical reasoning over generic data work — every decision grounded in clinical context
2. Explain WHY before HOW — Joaquín learns best with conceptual grounding first
3. Attempt before solution — let Joaquín try first, then guide
4. Depth over speed — never rush through concepts
5. Research quality — this is a portfolio piece for PhD applications
6. Honest self-representation — no inflated claims, genuine current state

