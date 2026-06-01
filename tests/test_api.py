import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

# Canonical high-risk patient from the Cleveland dataset (row 0).
# 63yo male, asymptomatic CP, oldpeak=2.3 — model predicts CAD.
HIGH_RISK = {
    "age": 63, "sex": 1, "cp": 3,
    "trestbps": 145, "chol": 233, "fbs": 1,
    "restecg": 0, "thalch": 150, "exang": 0,
    "oldpeak": 2.3, "slope": 0
}

# Low-risk profile: young female, high exercise capacity, no stress-test flags.
LOW_RISK = {
    "age": 35, "sex": 0, "cp": 2,
    "trestbps": 110, "chol": 180, "fbs": 0,
    "restecg": 0, "thalch": 190, "exang": 0,
    "oldpeak": 0.0, "slope": 2
}


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_high_risk_patient_predicts_cad():
    response = client.post("/predict", json=HIGH_RISK)
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] == 1
    assert body["probability"] > 0.5
    assert body["risk_level"] == "High"


def test_low_risk_patient_predicts_no_cad():
    response = client.post("/predict", json=LOW_RISK)
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] == 0
    assert body["probability"] < 0.5


def test_missing_field_returns_422():
    # Omitting 'slope' — Pydantic must reject the request before it reaches the model.
    incomplete = {k: v for k, v in HIGH_RISK.items() if k != "slope"}
    response = client.post("/predict", json=incomplete)
    assert response.status_code == 422


def test_out_of_range_chol_returns_422():
    # chol=0 is physiologically impossible — EDA excluded it as data entry error.
    # Pydantic Field(ge=100) must reject it.
    invalid = {**HIGH_RISK, "chol": 0}
    response = client.post("/predict", json=invalid)
    assert response.status_code == 422


def test_out_of_range_oldpeak_returns_422():
    # oldpeak < -2.0 was excluded in EDA as implausible.
    invalid = {**HIGH_RISK, "oldpeak": -3.0}
    response = client.post("/predict", json=invalid)
    assert response.status_code == 422
