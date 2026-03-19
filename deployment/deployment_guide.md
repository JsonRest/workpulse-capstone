# 🚀 Deployment Guide — WorkPulse

## Local Deployment (FastAPI)

### Prerequisites
- Python 3.9+
- Trained model in `models/best_model.pkl` (run Step 4 notebook first)

### Steps

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the model (if not already done)
python -m src.train --model xgboost --tune --output models/

# 3. Start the API
cd deployment
uvicorn app:app --reload --port 8000
```

### Test the API

```bash
# Health check
curl http://localhost:8000/health

# Predict
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
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
    "age_group": 0
  }'
```

### Swagger UI

Visit `http://localhost:8000/docs` for interactive API documentation.

---

## Docker Deployment

```bash
# Build
docker build -t workpulse-api -f deployment/Dockerfile .

# Run
docker run -p 8000:8000 workpulse-api

# Test
curl http://localhost:8000/health
```

---

## MLOps Practices

### Reproducible Environments
- `requirements.txt` — pinned Python dependencies
- `Dockerfile` — containerised environment
- `RANDOM_STATE = 42` — deterministic results

### Config-Driven Training
```bash
python -m src.train --model xgboost --tune --n-iter 30 --output models/
python -m src.train --model lightgbm --output models/
```

### Experiment Tracking
For production, integrate with MLflow or Weights & Biases:
```python
import mlflow
mlflow.set_experiment("workpulse")
with mlflow.start_run():
    mlflow.log_params(best_params)
    mlflow.log_metrics({"f1": 0.9062, "auc": 0.9868})
    mlflow.sklearn.log_model(model, "xgboost_tuned")
```

### CI/CD
See `.github/workflows/ci.yml` — runs lint (flake8) and tests (pytest) on every push.

### Monitoring Plan
1. Track prediction distribution drift (monthly)
2. Compare live predictions vs actual outcomes (quarterly)
3. Re-train if F1 drops below 0.85 on validation set
4. Re-audit fairness metrics after each retraining

### Versioning & Rollback
- Models saved with timestamp: `models/xgboost_v1.0_20260320.pkl`
- Previous model always kept as `models/best_model_prev.pkl`
- Rollback: swap symlink or environment variable `MODEL_PATH`
