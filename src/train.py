"""WorkPulse Model Training — config-driven."""
import numpy as np, joblib, json, time, os, argparse
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, classification_report
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from xgboost import XGBClassifier
from scipy.stats import randint, uniform
from src.data_pipeline import generate_dataset, split_and_scale, FEATURE_COLS

RANDOM_STATE = 42

def train_xgboost(tune=False, n_iter=30):
    df = generate_dataset()
    X_train, X_test, y_train, y_test, _, _, scaler = split_and_scale(df)
    if tune:
        print(f"Tuning XGBoost ({n_iter} iter x 3-fold CV)...")
        search = RandomizedSearchCV(
            XGBClassifier(eval_metric='logloss',random_state=RANDOM_STATE,verbosity=0),
            {'n_estimators':randint(200,500),'max_depth':randint(4,9),'learning_rate':uniform(0.03,0.2),
             'subsample':uniform(0.7,0.3),'colsample_bytree':uniform(0.6,0.4)},
            n_iter=n_iter, cv=StratifiedKFold(3,shuffle=True,random_state=RANDOM_STATE),
            scoring='f1', random_state=RANDOM_STATE, n_jobs=-1, refit=True)
        search.fit(X_train, y_train)
        model = search.best_estimator_
        print(f"Best CV F1: {search.best_score_:.4f}")
    else:
        model = XGBClassifier(n_estimators=300,max_depth=6,learning_rate=0.1,subsample=0.8,
            colsample_bytree=0.8,reg_alpha=0.1,reg_lambda=1.0,eval_metric='logloss',
            random_state=RANDOM_STATE,verbosity=0)
        model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]
    print(f"F1={f1_score(y_test,y_pred):.4f} AUC={roc_auc_score(y_test,y_prob):.4f}")
    return model, scaler

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--output", default="models/")
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    model, scaler = train_xgboost(tune=args.tune)
    joblib.dump(model, os.path.join(args.output, "best_model.pkl"))
    joblib.dump(scaler, os.path.join(args.output, "scaler.pkl"))
    model.save_model(os.path.join(args.output, "model.bst"))
    joblib.dump(FEATURE_COLS, os.path.join(args.output, "feature_columns.pkl"))
    print(f"Saved to {args.output}")
