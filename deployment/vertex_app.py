"""
WorkPulse — Vertex AI Custom Prediction Container
==================================================
Conforms to Vertex AI's custom container contract:
  - Health check at GET /health
  - Predictions at POST /predict

Run locally:
    python vertex_app.py

Test locally:
    curl http://localhost:8080/health
    curl -X POST http://localhost:8080/predict \
      -H "Content-Type: application/json" \
      -d '{"instances": [[0.75, 0.25, 0.7, -0.3, 1, 1, 0.3, 0.2, 7.5, 3000, 2, 27, 0]]}'
"""

import os
import numpy as np
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI(title="WorkPulse Vertex AI", version="1.0.0")

# ── Feature names (must match training order) ─────────────────
FEATURE_NAMES = [
    "overtime_index", "wellbeing_composite", "workload_pressure",
    "satisfaction_gap", "high_stress_flag", "tenure_risk_flag",
    "job_satisfaction", "work_life_balance", "log_income",
    "monthly_income", "tenure_years", "age", "age_group",
]

# ── Load model ─────────────────────────────────────────────────
model = None
model_path = os.environ.get("MODEL_PATH", "models/best_model.pkl")
if os.path.exists(model_path):
    model = joblib.load(model_path)
    print(f"Model loaded from {model_path}")
else:
    print(f"WARNING: Model not found at {model_path}. Predictions will fail.")


# ── Schemas ───────────────────────────────────────────────────
class PredictRequest(BaseModel):
    """Vertex AI sends instances as a list of feature arrays."""
    instances: List[List[float]]


class SinglePrediction(BaseModel):
    burnout_risk: int
    burnout_probability: float
    risk_level: str
    top_factors: Optional[List[dict]] = None


class PredictResponse(BaseModel):
    predictions: List[dict]


# ── Endpoints ─────────────────────────────────────────────────
@app.get("/health")
def health():
    """Health check — required by Vertex AI."""
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    """Prediction endpoint — required by Vertex AI.

    Expects:
        {"instances": [[0.75, 0.25, 0.7, ...], [0.1, 0.85, 0.1, ...]]}

    Each instance is a list of 13 floats in FEATURE_NAMES order.
    """
    results = []
    for instance in request.instances:
        features = np.array([instance])
        pred = int(model.predict(features)[0])
        prob = float(model.predict_proba(features)[0, 1])

        if prob < 0.3:
            risk = "Low"
        elif prob < 0.6:
            risk = "Medium"
        else:
            risk = "High"

        # Top 5 contributing factors
        top_factors = None
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            top_idx = np.argsort(importances)[::-1][:5]
            top_factors = [
                {"feature": FEATURE_NAMES[i], "importance": round(float(importances[i]), 4)}
                for i in top_idx
            ]

        results.append({
            "burnout_risk": pred,
            "burnout_probability": round(prob, 4),
            "risk_level": risk,
            "top_factors": top_factors,
        })

    return {"predictions": results}


# ── Run ───────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("AIP_HTTP_PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port)
