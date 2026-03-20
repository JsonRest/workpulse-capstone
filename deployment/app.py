"""WorkPulse API — FastAPI local deployment."""
import os, sys, numpy as np, joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Optional, List
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

app = FastAPI(title="WorkPulse API", version="1.0.0")

MODEL_PATH = os.environ.get("MODEL_PATH", "models/best_model.pkl")
SCALER_PATH = os.environ.get("SCALER_PATH", "models/scaler.pkl")
FEATURES_PATH = os.environ.get("FEATURES_PATH", "models/feature_columns.pkl")

model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
scaler = joblib.load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None
features = joblib.load(FEATURES_PATH) if os.path.exists(FEATURES_PATH) else None

class EmployeeInput(BaseModel):
    overtime_index: float = Field(ge=0, le=1)
    wellbeing_composite: float = Field(ge=0, le=1)
    workload_pressure: float = Field(ge=0, le=1)
    satisfaction_gap: float = Field(ge=-1, le=1)
    high_stress_flag: int = Field(ge=0, le=1)
    tenure_risk_flag: int = Field(ge=0, le=1)
    job_satisfaction: float = Field(ge=0, le=1)
    work_life_balance: float = Field(ge=0, le=1)
    log_income: float = Field(ge=5, le=12)
    monthly_income: float = Field(ge=1000)
    tenure_years: float = Field(ge=0, le=40)
    age: float = Field(ge=18, le=70)
    age_group: int = Field(ge=0, le=3)

class PredictionResponse(BaseModel):
    burnout_risk: int
    burnout_probability: float
    risk_level: str
    top_factors: Optional[List[dict]] = None

@app.get("/")
def root():
    return {"service": "WorkPulse API", "version": "1.0.0", "model_loaded": model is not None}

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictionResponse)
def predict(employee: EmployeeInput):
    feats = np.array([[employee.overtime_index, employee.wellbeing_composite,
        employee.workload_pressure, employee.satisfaction_gap,
        employee.high_stress_flag, employee.tenure_risk_flag,
        employee.job_satisfaction, employee.work_life_balance,
        employee.log_income, employee.monthly_income,
        employee.tenure_years, employee.age, employee.age_group]])
    pred = int(model.predict(feats)[0])
    prob = float(model.predict_proba(feats)[0, 1])
    risk = "Low" if prob < 0.3 else ("Medium" if prob < 0.6 else "High")
    top_factors = None
    if hasattr(model, "feature_importances_") and features:
        imp = model.feature_importances_
        top_idx = np.argsort(imp)[::-1][:5]
        top_factors = [{"feature": features[i], "importance": round(float(imp[i]),4)} for i in top_idx]
    return PredictionResponse(burnout_risk=pred, burnout_probability=round(prob,4),
                              risk_level=risk, top_factors=top_factors)
