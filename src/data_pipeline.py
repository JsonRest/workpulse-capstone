"""WorkPulse Data Pipeline — data loading, cleaning, feature engineering."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib, os, argparse

RANDOM_STATE = 42
FEATURE_COLS = [
    "overtime_index","wellbeing_composite","workload_pressure","satisfaction_gap",
    "high_stress_flag","tenure_risk_flag","job_satisfaction","work_life_balance",
    "log_income","monthly_income","tenure_years","age","age_group",
]
TARGET_COL = "burnout_risk"

def generate_dataset(n=44220, seed=RANDOM_STATE):
    np.random.seed(seed)
    ot=np.clip(np.random.beta(2,5,n)*1.2,0,1); wb=np.clip(np.random.normal(0.55,0.15,n),0,1)
    wp=np.clip(np.random.beta(2,5,n)*1.5,0,1); sg=np.clip(np.random.normal(0,0.22,n),-1,1)
    sf=np.random.choice([0,1],n,p=[0.62,0.38]).astype(float)
    tr=np.random.choice([0,1],n,p=[0.58,0.42]).astype(float)
    js=np.clip(np.random.normal(0.50,0.20,n),0,1); wlb=np.clip(np.random.normal(0.55,0.18,n),0,1)
    li=np.clip(np.random.normal(8.5,0.85,n),5,12); mi=np.clip(np.random.lognormal(8.5,0.80,n),1000,200000)
    ten=np.clip(np.random.exponential(5,n),0,40).astype(float)
    age_vals=np.random.randint(22,60,n).astype(float)
    ag=np.random.choice([0,1,2,3],n,p=[0.30,0.35,0.25,0.10]).astype(float)
    df = pd.DataFrame({"overtime_index":ot,"wellbeing_composite":wb,"workload_pressure":wp,
        "satisfaction_gap":sg,"high_stress_flag":sf,"tenure_risk_flag":tr,
        "job_satisfaction":js,"work_life_balance":wlb,"log_income":li,
        "monthly_income":mi,"tenure_years":ten,"age":age_vals,"age_group":ag})
    burnout_score = (
        0.18*np.where(ot>0.35,(ot-0.35)**2*10+0.3*ot,ot*0.5)+0.22*sf*(1-wb)**1.5+
        0.12*wp*(1-js)+0.08*(1-wb)+0.06*sf+0.05*tr+
        0.06*(np.exp(-0.5*((ten-2)/1.5)**2)+0.4*np.exp(-0.5*((ten-17)/4)**2))+
        0.04*np.tanh(-sg*2.5)+0.05*ot*np.where(age_vals<35,1.4,0.7)-0.06*sf*js+0.08*np.random.rand(n))
    df[TARGET_COL] = (burnout_score >= np.percentile(burnout_score, 65)).astype(int)
    return df

def split_and_scale(df, test_size=0.20, seed=RANDOM_STATE):
    X = df[FEATURE_COLS].values; y = df[TARGET_COL].values
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=test_size,stratify=y,random_state=seed)
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train); X_test_sc = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test, X_train_sc, X_test_sc, scaler

if __name__ == "__main__":
    df = generate_dataset()
    print(f"Dataset: {df.shape}, positive: {df[TARGET_COL].mean()*100:.1f}%")
