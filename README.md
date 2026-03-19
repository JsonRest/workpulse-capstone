# 🔥 WorkPulse: AI-Powered Employee Burnout Early Warning System

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> Predicting employee burnout before it costs you your best people.

---

## 📋 Project Overview

**WorkPulse** is an end-to-end machine learning system that predicts employee burnout risk using HR survey data, work patterns, and wellbeing indicators. It enables HR teams to proactively intervene before burnout leads to attrition, absenteeism, or productivity loss.

| Attribute | Value |
|-----------|-------|
| **Task Type** | Binary Classification |
| **Target** | `burnout_risk ∈ {0, 1}` |
| **Best Model** | XGBoost (Tuned) |
| **F1 Score** | 0.9062 |
| **AUC** | 0.9868 |
| **Recall** | 0.8969 |
| **Dataset** | 44,220 employees (3 Kaggle sources, unified) |

---

## 🎯 Problem Statement

Employee burnout costs organisations an estimated **$322B annually** through turnover, absenteeism, and lost productivity. Yet most companies detect burnout only after employees resign. WorkPulse uses machine learning to identify at-risk employees **weeks before** symptoms escalate, enabling targeted, compassionate HR interventions.

**Key question:** *Can we predict which employees are at high risk of burnout using HR survey data, work patterns, and wellbeing indicators?*

---

## 📊 Dataset

Three publicly available Kaggle datasets were unified into a single schema:

| Dataset | Rows | Features | Source |
|---------|------|----------|--------|
| IBM HR Analytics | 1,470 | 35 | [Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset) |
| Employee Burnout Analysis | 22,750 | 9 | [Kaggle](https://www.kaggle.com/datasets/blurredmachine/are-your-employees-burning-out) |
| Workplace Stress Survey | 20,000 | 15 | [Kaggle](https://www.kaggle.com/datasets/waqi786/workplace-stress-and-mental-health-dataset) |
| **Unified** | **44,220** | **22** | Schema alignment across all 3 |

**Target variable:** Composite burnout risk score thresholded at the 65th percentile (~35% positive class).

---

## 🏗️ Project Structure

```
workpulse-capstone/
├── README.md                          # This file
├── LICENSE                            # MIT License
├── requirements.txt                   # Python dependencies
├── Dockerfile                         # Reproducible environment
├── .gitignore                         # Git ignore rules
│
├── notebooks/                         # Jupyter notebooks (Steps 1-5)
│   ├── 01_problem_framing.ipynb       # Step 1: Problem definition & metrics
│   ├── 02_data_collection.ipynb       # Step 2: Data loading & profiling
│   ├── 03_eda_feature_engineering.ipynb# Step 3: EDA, cleaning, feature engineering
│   ├── 04_model_implementation.ipynb  # Step 4: Model training & comparison
│   └── 05_ethical_ai_bias.ipynb       # Step 5: SHAP, LIME, fairness audit
│
├── src/                               # Modular Python scripts
│   ├── __init__.py
│   ├── data_pipeline.py               # Data loading, cleaning, feature engineering
│   ├── train.py                       # Model training (config-driven)
│   └── predict.py                     # Inference API
│
├── data/
│   ├── raw/                           # Original CSVs (gitignored, see notebooks to generate)
│   └── processed/                     # Cleaned datasets, model comparison CSVs
│
├── models/                            # Saved model artefacts (.pkl, .keras)
│
├── reports/                           # Presentation decks
│   ├── WorkPulse_Business_Deck.pptx   # Executive presentation (12 slides)
│   ├── WorkPulse_Technical_Beamer.pdf # Technical presentation (LaTeX Beamer, 11 slides)
│   └── WorkPulse_Technical_Slides.ipynb # Technical presentation (Jupyter RISE)
│
├── docs/
│   ├── data_dictionary.md             # Full feature documentation
│   └── genai_usage.md                 # Generative AI usage documentation
│
├── deployment/                        # Deployment artefacts (Step 8)
│   ├── app.py                         # FastAPI application
│   ├── Dockerfile                     # Container for deployment
│   └── deployment_guide.md            # How to deploy
│
├── tests/                             # Unit tests
│   └── test_data_pipeline.py
│
├── plots/                             # Generated visualisations
│
└── .github/
    └── workflows/
        └── ci.yml                     # CI: lint + test
```

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/workpulse-capstone.git
cd workpulse-capstone
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run notebooks

Open the notebooks in order (01 → 05) in Jupyter or Google Colab. Each notebook is **self-contained** — it generates its own data, so you don't need external downloads.

```bash
jupyter notebook notebooks/
```

### 4. Run the API (Step 8)

```bash
cd deployment
uvicorn app:app --reload --port 8000
```

Then visit `http://localhost:8000/docs` for the Swagger UI.

---

## 📈 Results Summary

### Model Comparison (11 models across 5 phases)

| Phase | Model | F1 | AUC |
|-------|-------|----|-----|
| Baseline | Dummy Classifier | 0.29 | 0.50 |
| Baseline | Logistic Regression | 0.82 | 0.95 |
| Tree | Decision Tree (d=8) | 0.85 | 0.95 |
| Ensemble | Random Forest (300) | 0.89 | 0.98 |
| **Ensemble** | **XGBoost (Tuned)** | **0.91** | **0.99** |
| Ensemble | LightGBM (Tuned) | 0.90 | 0.99 |
| Deep | MLP Neural Network | 0.89 | 0.98 |
| Meta | Stacking (RF+XGB+LGBM→LR) | 0.89 | 0.98 |

### SHAP Feature Importance (Top 6)

1. `high_stress_flag` — 3.04
2. `overtime_index` — 2.80
3. `tenure_risk_flag` — 1.52
4. `wellbeing_composite` — 1.44
5. `tenure_years` — 1.22
6. `satisfaction_gap` — 0.96

### Fairness Audit — All Pass ✅

| Metric | Gender | Age | Income | Status |
|--------|--------|-----|--------|--------|
| Demographic Parity (>0.80) | 0.94 | 0.91 | 0.88 | ✅ |
| Equalized Odds (gap <0.10) | 0.02 | 0.04 | 0.06 | ✅ |
| Disparate Impact (>0.80) | 0.94 | 0.91 | 0.88 | ✅ |

---

## 🛠️ Tech Stack

- **Python 3.9+** — Core language
- **scikit-learn** — Preprocessing, model selection, evaluation
- **XGBoost / LightGBM** — Gradient boosting models
- **TensorFlow / Keras** — MLP neural network
- **SHAP / LIME** — Model explainability
- **matplotlib / seaborn** — Visualisation
- **FastAPI** — Model deployment API
- **Docker** — Reproducible environments

---

## 📝 Capstone Steps

| Step | Deliverable | Notebook / File |
|------|------------|-----------------|
| 1 | Problem framing + metrics | `notebooks/01_problem_framing.ipynb` |
| 2 | Data collection + dictionary | `notebooks/02_data_collection.ipynb` |
| 3 | EDA + feature engineering | `notebooks/03_eda_feature_engineering.ipynb` |
| 4 | Model training + comparison | `notebooks/04_model_implementation.ipynb` |
| 5 | Ethical AI + bias audit | `notebooks/05_ethical_ai_bias.ipynb` |
| 6 | Presentations (tech + business) | `reports/` |
| 7 | GitHub repo (this!) | You're looking at it |
| 8 | Deployment + MLOps | `deployment/` |
| 9 | Generative AI usage | `docs/genai_usage.md` |

---

## ⚠️ Limitations

- **Target leakage:** Composite target derived from input features (production would use independent HR outcomes)
- **Synthetic data:** Augmented from 44K to 500K rows; real-world performance may differ
- **Cross-sectional:** No temporal dimension; cannot capture burnout progression
- **Limited fairness scope:** Audited gender, age, income only; race/disability unavailable

See `notebooks/05_ethical_ai_bias.ipynb` for the full limitations analysis.

---

## 👤 Author

**Jesse Liamzon**
Post Graduate Programme in AI & Machine Learning

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
