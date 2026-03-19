"""
WorkPulse Prediction
====================
Load a trained model and make predictions on new employee data.

Usage:
    python -m src.predict --model models/best_model.pkl --input data/new_employees.csv
"""

import numpy as np
import pandas as pd
import joblib
import argparse
import json

from src.data_pipeline import FEATURE_COLS


def load_model(model_path: str, scaler_path: str = "models/scaler.pkl"):
    """Load a trained model and scaler."""
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


def predict(model, scaler, df: pd.DataFrame, use_scaled: bool = False):
    """Make predictions on a DataFrame.

    Parameters
    ----------
    model : fitted sklearn/xgboost model
    scaler : fitted StandardScaler
    df : pd.DataFrame with columns matching FEATURE_COLS
    use_scaled : bool
        Whether to scale input (True for LogReg/SVM/KNN, False for tree-based)

    Returns
    -------
    pd.DataFrame
        Original data with added columns: burnout_risk_pred, burnout_prob
    """
    X = df[FEATURE_COLS].values

    if use_scaled:
        X = scaler.transform(X)

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else None

    result = df.copy()
    result["burnout_risk_pred"] = predictions
    if probabilities is not None:
        result["burnout_prob"] = np.round(probabilities, 4)
        result["risk_level"] = pd.cut(
            probabilities,
            bins=[0, 0.3, 0.6, 1.0],
            labels=["Low", "Medium", "High"],
        )

    return result


def main():
    parser = argparse.ArgumentParser(description="WorkPulse prediction")
    parser.add_argument("--model", type=str, default="models/best_model.pkl")
    parser.add_argument("--scaler", type=str, default="models/scaler.pkl")
    parser.add_argument("--input", type=str, required=True, help="CSV with employee data")
    parser.add_argument("--output", type=str, default=None, help="Output CSV path")
    args = parser.parse_args()

    model, scaler = load_model(args.model, args.scaler)
    df = pd.read_csv(args.input)

    print(f"Loaded {len(df)} employees from {args.input}")

    result = predict(model, scaler, df, use_scaled=False)

    n_high = (result["burnout_risk_pred"] == 1).sum()
    print(f"\nPredictions:")
    print(f"  High risk: {n_high} ({n_high/len(result)*100:.1f}%)")
    print(f"  Low risk:  {len(result)-n_high} ({(len(result)-n_high)/len(result)*100:.1f}%)")

    if "risk_level" in result.columns:
        print(f"\nRisk distribution:")
        print(result["risk_level"].value_counts().to_string())

    if args.output:
        result.to_csv(args.output, index=False)
        print(f"\nSaved predictions to {args.output}")
    else:
        print(f"\nTop 10 highest risk:")
        print(result.nlargest(10, "burnout_prob")[
            FEATURE_COLS[:5] + ["burnout_prob", "risk_level"]
        ].to_string(index=False))


if __name__ == "__main__":
    main()
