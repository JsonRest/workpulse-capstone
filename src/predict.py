"""WorkPulse Prediction — load model and predict."""
import numpy as np, pandas as pd, joblib, argparse
from src.data_pipeline import FEATURE_COLS

def predict(model, df, feature_cols=FEATURE_COLS):
    X = df[feature_cols].values
    preds = model.predict(X)
    probs = model.predict_proba(X)[:,1] if hasattr(model,'predict_proba') else None
    result = df.copy()
    result['burnout_risk_pred'] = preds
    if probs is not None:
        result['burnout_prob'] = np.round(probs, 4)
        result['risk_level'] = pd.cut(probs, bins=[0,0.3,0.6,1.0], labels=['Low','Medium','High'])
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/best_model.pkl")
    parser.add_argument("--input", required=True)
    args = parser.parse_args()
    model = joblib.load(args.model)
    df = pd.read_csv(args.input)
    result = predict(model, df)
    print(f"High risk: {(result['burnout_risk_pred']==1).sum()}/{len(result)}")
