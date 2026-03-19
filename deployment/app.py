"""
WorkPulse API — FastAPI Deployment
==================================
REST API for employee burnout risk prediction.

Run:
    uvicorn deployment.app:app --reload --port 8000

Docs:
    http://localhost:8000/docs
"""

import os
import sys
import numpy as np
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

app = FastAPI(
    title="WorkPulse API",
    description="AI-Powered Employee Burnout Early Warning System",
    version="1.0.0",
)

# ── Load model on startup ─────────────────────────────────────
MODEL_PATH = os.environ.get("MODEL_PATH", "models/best_model.pkl")
SCALER_PATH = os.environ.get("SCALER_PATH", "models/scaler.pkl")

model = None
scaler = None


@app.on_event("startup")
def load_artifacts():
    global model, scaler
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        print(f"Model loaded from {MODEL_PATH}")
    except FileNotFoundError:
        print(f"WARNING: Model not found at {MODEL_PATH}. Run training first.")
        print("  python -m src.train --model xgboost --tune --output models/")


# ── Request / Response schemas ────────────────────────────────
class EmployeeInput(BaseModel):
    """Single employee feature vector for prediction."""
    overtime_index: float = Field(ge=0, le=1, description="Normalised overtime intensity")
    wellbeing_composite: float = Field(ge=0, le=1, description="Holistic wellbeing score")
    workload_pressure: float = Field(ge=0, le=1, description="Compounded workload signal")
    satisfaction_gap: float = Field(ge=-1, le=1, description="WLB minus job satisfaction")
    high_stress_flag: int = Field(ge=0, le=1, description="1 if stress >= 7/10")
    tenure_risk_flag: int = Field(ge=0, le=1, description="1 if 1-3yr or 7-9yr tenure")
    job_satisfaction: float = Field(ge=0, le=1, description="Normalised job satisfaction")
    work_life_balance: float = Field(ge=0, le=1, description="Normalised WLB rating")
    log_income: float = Field(ge=5, le=12, description="Log-transformed monthly income")
    monthly_income: float = Field(ge=1000, description="Monthly income (raw)")
    tenure_years: float = Field(ge=0, le=40, description="Years at company")
    age: float = Field(ge=18, le=70, description="Employee age")
    age_group: int = Field(ge=0, le=3, description="0=Under30, 1=30-39, 2=40-49, 3=50+")

    class Config:
        json_schema_extra = {
            "example": {
                "overtime_index": 0.6,
                "wellbeing_composite": 0.35,
                "workload_pressure": 0.5,
                "satisfaction_gap": -0.2,
                "high_stress_flag": 1,
                "tenure_risk_flag": 1,
                "job_satisfaction": 0.4,
                "work_life_balance": 0.3,
                "log_income": 8.2,
                "monthly_income": 5000,
                "tenure_years": 2,
                "age": 28,
                "age_group": 0,
            }
        }


class PredictionResponse(BaseModel):
    burnout_risk: int = Field(description="0=Low risk, 1=High risk")
    burnout_probability: float = Field(description="Probability of burnout (0-1)")
    risk_level: str = Field(description="Low / Medium / High")
    top_factors: Optional[list] = Field(description="Top contributing factors")


# ── Endpoints ─────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "service": "WorkPulse API",
        "version": "1.0.0",
        "status": "running",
        "model_loaded": model is not None,
    }


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(employee: EmployeeInput):
    """Predict burnout risk for a single employee."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run training first.")

    # Build feature vector
    features = np.array([[
        employee.overtime_index, employee.wellbeing_composite,
        employee.workload_pressure, employee.satisfaction_gap,
        employee.high_stress_flag, employee.tenure_risk_flag,
        employee.job_satisfaction, employee.work_life_balance,
        employee.log_income, employee.monthly_income,
        employee.tenure_years, employee.age, employee.age_group,
    ]])

    # Predict
    prediction = int(model.predict(features)[0])
    probability = float(model.predict_proba(features)[0, 1])

    # Risk level
    if probability < 0.3:
        risk_level = "Low"
    elif probability < 0.6:
        risk_level = "Medium"
    else:
        risk_level = "High"

    # Top factors (feature importance from model)
    feature_names = [
        "overtime_index", "wellbeing_composite", "workload_pressure",
        "satisfaction_gap", "high_stress_flag", "tenure_risk_flag",
        "job_satisfaction", "work_life_balance", "log_income",
        "monthly_income", "tenure_years", "age", "age_group",
    ]
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        top_idx = np.argsort(importances)[::-1][:5]
        top_factors = [
            {"feature": feature_names[i], "importance": round(float(importances[i]), 4)}
            for i in top_idx
        ]
    else:
        top_factors = None

    return PredictionResponse(
        burnout_risk=prediction,
        burnout_probability=round(probability, 4),
        risk_level=risk_level,
        top_factors=top_factors,
    )
