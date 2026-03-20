# Data Dictionary — WorkPulse

## Target
| Variable | Type | Values | Description |
|----------|------|--------|-------------|
| `burnout_risk` | Binary | 0, 1 | 1 = high burnout risk (>=65th percentile composite) |

## Model Features (13)
| Variable | Type | Range | Description |
|----------|------|-------|-------------|
| `overtime_index` | Continuous | [0,1] | Normalised overtime intensity |
| `wellbeing_composite` | Continuous | [0,1] | Avg of satisfaction, WLB, sleep, activity |
| `workload_pressure` | Continuous | [0,1] | overtime_index x (1 - WLB) |
| `satisfaction_gap` | Continuous | [-1,1] | WLB - job_satisfaction |
| `high_stress_flag` | Binary | 0,1 | 1 if stress >= 7/10 |
| `tenure_risk_flag` | Binary | 0,1 | 1 if 1-3yr or 7-9yr tenure |
| `job_satisfaction` | Continuous | [0,1] | Normalised job satisfaction |
| `work_life_balance` | Continuous | [0,1] | Normalised WLB rating |
| `log_income` | Continuous | [5,12] | log1p(monthly_income) |
| `monthly_income` | Continuous | [1000,200000] | Raw monthly income |
| `tenure_years` | Integer | [0,40] | Years at company |
| `age` | Integer | [22,60] | Employee age |
| `age_group` | Ordinal | 0-3 | Under30, 30-39, 40-49, 50+ |

## Sensitive Attributes (fairness audit only)
| Variable | Values | Description |
|----------|--------|-------------|
| `gender` | 0=Male, 1=Female | Self-reported |
| `age_bracket` | 0-3 | Under30, 30-40, 40-50, 50+ |
| `income_bracket` | 0-2 | Low, Medium, High |
