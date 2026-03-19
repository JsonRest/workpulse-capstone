# 📖 Data Dictionary — WorkPulse

## Target Variable

| Variable | Type | Values | Description |
|----------|------|--------|-------------|
| `burnout_risk` | Binary | 0, 1 | **Target.** 1 = high burnout risk (≥65th percentile composite score). Derived from weighted combination of stress, overtime, wellbeing, and satisfaction indicators. |

## Model Features (13 selected by consensus)

### Engineered Features (8)

| Variable | Type | Range | Description | Justification |
|----------|------|-------|-------------|---------------|
| `overtime_index` | Continuous | [0, 1] | Normalised overtime intensity: hours/max + overtime flag weight | WHO identifies overtime as primary burnout driver |
| `wellbeing_composite` | Continuous | [0, 1] | Average of job satisfaction, WLB, sleep quality, physical activity (each normalised 0-1) | Holistic wellbeing proxy capturing multiple dimensions |
| `workload_pressure` | Continuous | [0, 1] | `overtime_index × (1 - work_life_balance)` — compounded stress signal | High hours + poor WLB = multiplicative risk |
| `satisfaction_gap` | Continuous | [-1, 1] | `work_life_balance - job_satisfaction` — aspiration vs reality mismatch | Large negative gap = unmet expectations |
| `high_stress_flag` | Binary | 0, 1 | 1 if self-reported stress level ≥ 7/10 | Clinical threshold for adverse outcomes |
| `tenure_risk_flag` | Binary | 0, 1 | 1 if tenure is 1-3 years OR 7-9 years | Known high-attrition windows in I/O psychology |
| `log_income` | Continuous | [5, 12] | `log₁ₚ(monthly_income)` — log-transformed income | Normalises right-skewed income distribution |
| `age_group` | Ordinal | 0, 1, 2, 3 | Binned age: 0=Under30, 1=30-39, 2=40-49, 3=50+ | Non-linear age effects on burnout |

### Original Features Retained (5)

| Variable | Type | Range | Description |
|----------|------|-------|-------------|
| `job_satisfaction` | Continuous | [0, 1] | Normalised job satisfaction score (0=low, 1=high) |
| `work_life_balance` | Continuous | [0, 1] | Normalised work-life balance rating |
| `monthly_income` | Continuous | [1000, 200000] | Raw monthly income in local currency |
| `tenure_years` | Integer | [0, 40] | Years at current company |
| `age` | Integer | [22, 60] | Employee age in years |

## Sensitive Attributes (NOT used as model features — for fairness audit only)

| Variable | Type | Values | Description |
|----------|------|--------|-------------|
| `gender` | Binary | 0=Male, 1=Female | Self-reported gender |
| `age_bracket` | Ordinal | 0-3 | Under30, 30-40, 40-50, 50+ |
| `income_bracket` | Ordinal | 0-2 | Low, Medium, High (tertile bins) |

## Data Sources

| Dataset | URL | Licence |
|---------|-----|---------|
| IBM HR Analytics Attrition | https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset | CC0 Public Domain |
| Employee Burnout Analysis | https://www.kaggle.com/datasets/blurredmachine/are-your-employees-burning-out | CC BY 4.0 |
| Workplace Stress Survey | https://www.kaggle.com/datasets/waqi786/workplace-stress-and-mental-health-dataset | CC0 Public Domain |
