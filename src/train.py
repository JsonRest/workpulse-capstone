"""
WorkPulse Model Training
========================
Config-driven model training with experiment tracking.

Usage:
    python -m src.train --model xgboost --output models/
"""

import numpy as np
import joblib
import time
import os
import argparse
import json

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from scipy.stats import randint, uniform

from src.data_pipeline import generate_dataset, split_and_scale, FEATURE_COLS

RANDOM_STATE = 42

# ── Model configurations ──────────────────────────────────────
MODEL_CONFIGS = {
    "logistic_regression": {
        "class": LogisticRegression,
        "params": {"max_iter": 1000, "random_state": RANDOM_STATE},
        "use_scaled": True,
    },
    "random_forest": {
        "class": RandomForestClassifier,
        "params": {
            "n_estimators": 300, "max_depth": 12, "min_samples_leaf": 10,
            "random_state": RANDOM_STATE, "n_jobs": -1,
        },
        "use_scaled": False,
    },
    "xgboost": {
        "class": XGBClassifier,
        "params": {
            "n_estimators": 300, "max_depth": 6, "learning_rate": 0.1,
            "subsample": 0.8, "colsample_bytree": 0.8,
            "reg_alpha": 0.1, "reg_lambda": 1.0,
            "eval_metric": "logloss", "random_state": RANDOM_STATE, "verbosity": 0,
        },
        "use_scaled": False,
        "tune_params": {
            "n_estimators": randint(200, 500),
            "max_depth": randint(4, 9),
            "learning_rate": uniform(0.03, 0.2),
            "subsample": uniform(0.7, 0.3),
            "colsample_bytree": uniform(0.6, 0.4),
            "min_child_weight": randint(1, 15),
            "reg_alpha": uniform(0, 1),
            "reg_lambda": uniform(0.5, 2.5),
        },
    },
    "lightgbm": {
        "class": LGBMClassifier,
        "params": {
            "n_estimators": 300, "max_depth": 8, "learning_rate": 0.1,
            "random_state": RANDOM_STATE, "verbose": -1,
        },
        "use_scaled": False,
    },
}


def train_model(model_name: str, tune: bool = False, n_iter: int = 30):
    """Train a model and return metrics.

    Parameters
    ----------
    model_name : str
        Key from MODEL_CONFIGS.
    tune : bool
        Whether to run RandomizedSearchCV.
    n_iter : int
        Number of tuning iterations.

    Returns
    -------
    tuple
        (fitted_model, metrics_dict)
    """
    config = MODEL_CONFIGS[model_name]

    # Generate data
    df = generate_dataset()
    X_train, X_test, y_train, y_test, X_train_sc, X_test_sc, scaler = split_and_scale(df)

    Xtr = X_train_sc if config["use_scaled"] else X_train
    Xte = X_test_sc if config["use_scaled"] else X_test

    if tune and "tune_params" in config:
        print(f"Tuning {model_name} ({n_iter} iterations × 3-fold CV)...")
        search = RandomizedSearchCV(
            config["class"](**{k: v for k, v in config["params"].items()
                               if k not in config["tune_params"]}),
            param_distributions=config["tune_params"],
            n_iter=n_iter,
            cv=StratifiedKFold(3, shuffle=True, random_state=RANDOM_STATE),
            scoring="f1",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            refit=True,
        )
        t0 = time.time()
        search.fit(Xtr, y_train)
        elapsed = time.time() - t0
        model = search.best_estimator_
        print(f"  Best CV F1: {search.best_score_:.4f}")
        print(f"  Best params: {search.best_params_}")
    else:
        print(f"Training {model_name}...")
        model = config["class"](**config["params"])
        t0 = time.time()
        model.fit(Xtr, y_train)
        elapsed = time.time() - t0

    # Evaluate
    y_pred = model.predict(Xte)
    y_prob = model.predict_proba(Xte)[:, 1] if hasattr(model, "predict_proba") else None

    metrics = {
        "model": model_name,
        "tuned": tune,
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "auc": round(roc_auc_score(y_test, y_prob), 4) if y_prob is not None else None,
        "training_time_s": round(elapsed, 2),
    }

    print(f"\nResults:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    print(f"\n{classification_report(y_test, y_pred, target_names=['No Burnout', 'Burnout Risk'])}")

    return model, metrics, scaler


def main():
    parser = argparse.ArgumentParser(description="WorkPulse model training")
    parser.add_argument("--model", type=str, default="xgboost",
                        choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--tune", action="store_true", help="Run hyperparameter tuning")
    parser.add_argument("--n-iter", type=int, default=30, help="Tuning iterations")
    parser.add_argument("--output", type=str, default="models/", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    model, metrics, scaler = train_model(args.model, tune=args.tune, n_iter=args.n_iter)

    # Save
    suffix = "_tuned" if args.tune else ""
    model_path = os.path.join(args.output, f"{args.model}{suffix}_model.pkl")
    scaler_path = os.path.join(args.output, "scaler.pkl")
    metrics_path = os.path.join(args.output, f"{args.model}{suffix}_metrics.json")

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved: {model_path}")
    print(f"Saved: {scaler_path}")
    print(f"Saved: {metrics_path}")


if __name__ == "__main__":
    main()
