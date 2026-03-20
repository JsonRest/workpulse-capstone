# 🔥 WorkPulse: AI-Powered Employee Burnout Early Warning System

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> Predicting employee burnout before it costs you your best people.
<img width="1713" height="822" alt="Workpulse Risk Assessment" src="https://github.com/user-attachments/assets/06dafc6f-fa16-4095-90d2-0cf96854ca2d" />
<img width="1711" height="795" alt="Workpulse AI Advisor" src="https://github.com/user-attachments/assets/68e4fb17-c96c-4457-b279-fa73e1ef5726" />
<img width="1709" height="814" alt="Workpulse EDA Summary" src="https://github.com/user-attachments/assets/b1273a53-8776-43fc-b305-c02494e12e8d" />

## 📋 Project Overview

**WorkPulse** is an end-to-end machine learning system that predicts employee burnout risk using HR survey data, work patterns, and wellbeing indicators.

| Attribute | Value |
|-----------|-------|
| **Task Type** | Binary Classification |
| **Target** | `burnout_risk ∈ {0, 1}` |
| **Best Model** | XGBoost (Tuned) |
| **F1 Score** | 0.9062 |
| **AUC** | 0.9868 |
| **Recall** | 0.8969 |
| **Dataset** | 44,220 employees (3 Kaggle sources, unified) |

## 🎯 Problem Statement

Employee burnout costs organisations an estimated **$322B annually**. WorkPulse uses ML to identify at-risk employees **before** symptoms escalate, enabling targeted HR interventions.

## 📊 Dataset

| Dataset | Rows | Features | Source |
|---------|------|----------|--------|
| IBM HR Analytics | 1,470 | 35 | [Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset) |
| Employee Burnout Analysis | 22,750 | 9 | [Kaggle](https://www.kaggle.com/datasets/blurredmachine/are-your-employees-burning-out) |
| Workplace Stress Survey | 20,000 | 15 | [Kaggle](https://www.kaggle.com/datasets/waqi786/workplace-stress-and-mental-health-dataset) |
| **Unified** | **44,220** | **22** | Schema alignment across all 3 |

## 🏗️ Project Structure

```
workpulse-capstone/
├── README.md
├── LICENSE
├── requirements.txt
├── Dockerfile
├── .gitignore
├── notebooks/
│   ├── 01_problem_framing.ipynb
│   ├── 02_data_collection.ipynb
│   ├── 03_eda_feature_engineering.ipynb
│   ├── 04_model_implementation.ipynb
│   ├── 05_ethical_ai_bias.ipynb
│   ├── 06_presentation_communication.ipynb
│   ├── 07_github_profile.ipynb
│   ├── 08_deployment_mlops.ipynb
│   └── 09_genai_usage.ipynb
├── src/
│   ├── __init__.py
│   ├── data_pipeline.py
│   ├── train.py
│   └── predict.py
├── data/raw/ & data/processed/
├── models/
├── reports/         (presentations: .pptx, .pdf, .tex, .ipynb)
├── docs/            (data_dictionary.md, genai_usage.md)
├── deployment/      (FastAPI app, Vertex AI files, Docker)
├── tests/
└── .github/workflows/ci.yml
```

## 🚀 Quick Start

```bash
git clone https://github.com/JsonRest/workpulse-capstone.git
cd workpulse-capstone
pip install -r requirements.txt
jupyter notebook notebooks/
```

## 📈 Results Summary

| Phase | Model | F1 | AUC |
|-------|-------|----|-----|
| Baseline | Logistic Regression | 0.82 | 0.95 |
| Ensemble | Random Forest | 0.89 | 0.98 |
| **Ensemble** | **XGBoost (Tuned)** | **0.91** | **0.99** |
| Deep | MLP Neural Network | 0.89 | 0.98 |

### SHAP Top Factors
1. `high_stress_flag` (3.04) 2. `overtime_index` (2.80) 3. `tenure_risk_flag` (1.52)

### Fairness — All Pass ✅
Demographic Parity, Equalized Odds, Disparate Impact across gender, age, income.

## 🚀 Deployment

**Local:** `uvicorn deployment.app:app --port 8000`
**Docker:** `docker build -t workpulse-api . && docker run -p 8000:8000 workpulse-api`
**Vertex AI:** See `deployment/deploy_vertex.sh`

## 📝 Capstone Steps

| Step | Deliverable | File |
|------|------------|------|
| 1 | Problem framing | `notebooks/01_problem_framing.ipynb` |
| 2 | Data collection | `notebooks/02_data_collection.ipynb` |
| 3 | EDA + features | `notebooks/03_eda_feature_engineering.ipynb` |
| 4 | Model training | `notebooks/04_model_implementation.ipynb` |
| 5 | Ethical AI | `notebooks/05_ethical_ai_bias.ipynb` |
| 6 | Presentations | `notebooks/06_presentation_communication.ipynb` + `reports/` |
| 7 | GitHub repo | `notebooks/07_github_profile.ipynb` + this repo |
| 8 | Deployment | `notebooks/08_deployment_mlops.ipynb` + `deployment/` |
| 9 | GenAI usage | `notebooks/09_genai_usage.ipynb` + `deployment/WorkPulse_GenAI_Advisor.jsx` |

## 👤 Author

**[Your Name]** — Post Graduate Programme in AI & Machine Learning

## 📄 License

MIT License — see [LICENSE](LICENSE).

---

## 🤖 GenAI-Enhanced Application (Step 9)

WorkPulse includes an **AI-powered burnout advisor** built with Claude API:

| Feature | Description |
|---------|-------------|
| **Risk Assessment + AI Insight** | XGBoost prediction + Claude-generated explanations and interventions |
| **AI Advisor Chat** | Multi-turn chatbot for HR managers to ask about risk, interventions, model details |
| **Auto-EDA Generator** | LLM-generated exploratory data analysis summaries from dataset statistics |

See `notebooks/07_genai_usage.ipynb` for documentation and `deployment/WorkPulse_GenAI_Advisor.jsx` for the app code.
