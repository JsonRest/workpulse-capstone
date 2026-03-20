# Generative AI Usage — WorkPulse

## Tool Used
Claude (Anthropic) — Code generation, architecture design, documentation.

## Usage by Step
| Step | AI Usage | Human Contribution |
|------|----------|-------------------|
| 1 | Problem statement refinement | Domain selection, KPI definition |
| 2 | Data loading code, dictionary template | Dataset selection, quality assessment |
| 3 | Preprocessing pipeline, EDA visualisations | Feature design rationale, threshold decisions |
| 4 | 11-model training pipeline, evaluation framework | Model selection validation in Colab |
| 5 | SHAP/LIME/PDP code, fairness metrics | Ethical framing, mitigation strategy selection |
| 6 | Presentation generation (PPTX + LaTeX) | Narrative flow, final review |
| 7 | Repo structure, README, CI config | Git operations, final review |
| 8 | FastAPI app, Vertex AI deployment, MLOps | Deployment execution, debugging |
| 9 | This document | Review and validation |

## Key Principle
All AI-generated code was reviewed, tested in Google Colab, and validated before inclusion.

## GenAI-Enhanced Application

The flagship Step 9 deliverable is the **WorkPulse AI Advisor** (`deployment/WorkPulse_GenAI_Advisor.jsx`), a React application combining:
- XGBoost burnout prediction (JS port)
- Claude API for personalised risk explanations
- Multi-turn AI advisor chatbot for HR managers
- Automated EDA summary generation

See `notebooks/07_genai_usage.ipynb` for full documentation, architecture, and prompt engineering details.
