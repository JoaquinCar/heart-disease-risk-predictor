import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.experimental import enable_iterative_imputer  # must import before IterativeImputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

# Path relative to project root — run this script from heart-risk-predictor/
df = pd.read_csv('data/heart_disease_processed.csv')

# Separate features from target label
X = df.drop('target', axis=1)
y = df['target']

# stratify=y ensures class ratio (55% CAD / 45% no-CAD) is preserved in both splits
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Best params from GridSearchCV in notebook (ROC-AUC CV = 0.895).
# Defined once here — reused in both RandomForestClassifier and mlflow.log_params()
# so changing a value only requires editing one place.
params = {
    'max_depth': 5,
    'min_samples_split': 10,
    'n_estimators': 200,
    'random_state': 42
}

# Pipeline chains: impute → model.
# OrdinalEncoder removed: heart_disease_processed.csv was already imputed in the EDA
# notebook — categorical columns contain fractional floats (e.g. restecg=0.951) that
# OrdinalEncoder rejects as unknown categories on the test split.
# IterativeImputer is kept: harmless on already-clean data, but needed later when
# the API receives new patient inputs that may have missing values.
pipeline = Pipeline([
    ('imputer', IterativeImputer(max_iter=10, random_state=42)),
    ('model', RandomForestClassifier(**params))  # ** unpacks dict as keyword arguments
])

# Everything inside start_run() gets tracked: params, metrics, model artifact.
# MLflow creates a local mlruns/ directory to store all run data.
with mlflow.start_run():

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]  # probability of CAD (class 1)

    # Log hyperparameters — these become searchable/comparable across runs
    mlflow.log_params({**params, 'test_size': 0.2, 'imputer_max_iter': 10})

    # Log evaluation metrics on the hold-out test set (184 patients)
    mlflow.log_metrics({
        'roc_auc': roc_auc_score(y_test, y_prob),
        'f1': f1_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred)
    })

    # Save the full pipeline as one artifact and register it in MLflow Model Registry.
    # registered_model_name="cad-risk-predictor" means the API can load it by name
    # instead of by Run ID — decouples the API from specific training runs.
    mlflow.sklearn.log_model(pipeline, "model", registered_model_name="cad-risk-predictor")

    print(f"ROC-AUC : {roc_auc_score(y_test, y_prob):.3f}")
    print(f"F1      : {f1_score(y_test, y_pred):.3f}")
    print(f"Run ID  : {mlflow.active_run().info.run_id}")
