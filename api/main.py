from fastapi import FastAPI
from pydantic import BaseModel, Field
import mlflow.sklearn
import pandas as pd

app = FastAPI(
    title="CAD Risk Predictor",
    description="Coronary Artery Disease risk prediction from clinical features."
)

# Model loaded once at startup and reused for every request.
# Loading inside predict() would reload from disk on every call — slow.
model = mlflow.sklearn.load_model("models:/cad-risk-predictor/1")


class PatientFeatures(BaseModel):
    """11 clinical features required for CAD prediction.
    Values must be numeric (not string labels) matching training encoding.
    """
    age: int = Field(ge=1, le=120)
    sex: int = Field(ge=0, le=1)            # 0=female, 1=male
    cp: int = Field(ge=0, le=3)             # 0=typical angina, 1=atypical, 2=non-anginal, 3=asymptomatic
    trestbps: float = Field(ge=50, le=250)  # resting blood pressure (mmHg)
    chol: float = Field(ge=100, le=700)     # serum cholesterol (mg/dl) — ge=100: chol=0 is physiologically impossible
    fbs: int = Field(ge=0, le=1)            # fasting blood sugar > 120 mg/dl
    restecg: int = Field(ge=0, le=2)        # resting ECG result
    thalch: float = Field(ge=60, le=220)    # maximum heart rate achieved
    exang: int = Field(ge=0, le=1)          # exercise induced angina
    oldpeak: float = Field(ge=-2.0, le=10)  # ST depression — ge=-2.0 matches EDA threshold
    slope: int = Field(ge=0, le=2)          # slope of peak exercise ST segment


class PredictionResponse(BaseModel):
    probability: float  # P(CAD) — continuous score between 0 and 1
    prediction: int     # 0=no CAD, 1=CAD
    risk_level: str     # illustrative label only, not a clinical threshold


def classify_risk(probability: float) -> str:
    # Thresholds are illustrative for portfolio purposes.
    # Clinical deployment requires calibration against outcome data.
    if probability < 0.3:
        return "Low"
    elif probability < 0.6:
        return "Moderate"
    return "High"


@app.get("/health")
def health():
    return {"status": "ok", "model": "cad-risk-predictor/1"}


@app.post("/predict", response_model=PredictionResponse)
def predict(patient: PatientFeatures):
    # Convert Pydantic model to single-row DataFrame.
    # DataFrame needed because the pipeline was trained on pandas DataFrames
    # and expects column names — a plain list would lose feature name order.
    df = pd.DataFrame([patient.model_dump()])

    probability = float(model.predict_proba(df)[0, 1])  # P(CAD)
    prediction = int(model.predict(df)[0])

    return PredictionResponse(
        probability=round(probability, 4),
        prediction=prediction,
        risk_level=classify_risk(probability)
    )
