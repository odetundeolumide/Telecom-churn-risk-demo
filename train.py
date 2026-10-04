"""Train the telecom churn model and write honest, correctly-oriented metrics.

Usage:  python train.py telecom_churn.csv

Differences from the notebook:
  - Stratified 80/20 split (the notebook's split was not stratified).
  - Metrics always use (y_true, y_pred) order. Recall = share of REAL churners caught.
  - Model is picked by cross-validation on the training set only.
  - The alert threshold is chosen on out-of-fold training predictions (never on the test set),
    because missing a churner costs more than a false alarm.
  - Model, threshold and metrics are saved together.
"""
import json
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             fbeta_score, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import (RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict,
                                     cross_val_score, train_test_split)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
FEATURES = ["AccountWeeks", "ContractRenewal", "DataPlan", "DataUsage", "CustServCalls",
            "DayMins", "DayCalls", "MonthlyCharge", "OverageFee", "RoamMins"]
TARGET = "Churn"

df = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "telecom_churn.csv")
df.columns = df.columns.str.strip()
missing = set(FEATURES + [TARGET]) - set(df.columns)
if missing:
    sys.exit(f"Missing columns: {sorted(missing)}")
n_dups = int(df.duplicated().sum())
print(f"rows={len(df)} duplicates={n_dups} missing_values={int(df.isnull().sum().sum())} churn_rate={df[TARGET].mean():.3f}")

X, y = df[FEATURES], df[TARGET]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)

candidates = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                            random_state=SEED, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(random_state=SEED),
}
cv_scores = {}
for name, mdl in candidates.items():
    auc = cross_val_score(mdl, Xtr, ytr, cv=cv, scoring="roc_auc").mean()
    ap = cross_val_score(mdl, Xtr, ytr, cv=cv, scoring="average_precision").mean()
    cv_scores[name] = {"cv_auc": round(float(auc), 4), "cv_avg_precision": round(float(ap), 4)}
    print(f"{name:20s} CV AUC={auc:.4f}  CV avg-precision={ap:.4f}")

best_name = max(cv_scores, key=lambda k: cv_scores[k]["cv_avg_precision"])
model = candidates[best_name]
print("selected:", best_name)

# Threshold chosen on out-of-fold TRAIN predictions, maximising F2 (recall counts double).
oof = cross_val_predict(model, Xtr, ytr, cv=cv, method="predict_proba")[:, 1]
grid = np.arange(0.05, 0.96, 0.01)
f2 = [fbeta_score(ytr, (oof >= t).astype(int), beta=2) for t in grid]
threshold = float(round(grid[int(np.argmax(f2))], 2))

model.fit(Xtr, ytr)
prob = model.predict_proba(Xte)[:, 1]


def at(t):
    p = (prob >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(yte, p).ravel()
    return {"threshold": t,
            "recall": round(float(recall_score(yte, p)), 3),
            "precision": round(float(precision_score(yte, p)), 3),
            "f1": round(float(f1_score(yte, p)), 3),
            "churners_caught": int(tp), "churners_missed": int(fn), "false_alarms": int(fp),
            "churners_in_test": int(tp + fn)}


# One 80/20 split is noisy with only ~100 churners in the test set, so also report the
# repeated cross-validation AUC over ALL the data. This is the number to quote.
rcv = cross_val_score(candidates[best_name], X, y, scoring="roc_auc", n_jobs=-1,
                      cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED))
print(f"repeated 5-fold CV AUC over all data: {rcv.mean():.3f} +/- {rcv.std():.3f}")

perm = permutation_importance(model, Xte, yte, scoring="average_precision",
                              n_repeats=10, random_state=SEED, n_jobs=-1)
importance = (pd.Series(perm.importances_mean, index=FEATURES).sort_values(ascending=False)
              .round(4).to_dict())

segments = {
    "no_contract_renewal": round(float(df[df.ContractRenewal == 0][TARGET].mean()), 3),
    "contract_renewed": round(float(df[df.ContractRenewal == 1][TARGET].mean()), 3),
    "4plus_service_calls": round(float(df[df.CustServCalls >= 4][TARGET].mean()), 3),
    "under_4_service_calls": round(float(df[df.CustServCalls < 4][TARGET].mean()), 3),
}

metrics = {
    "rows": len(df), "duplicates": n_dups, "churn_rate": round(float(df[TARGET].mean()), 4),
    "test_rows": len(yte), "selected_model": best_name, "candidates_cv": cv_scores,
    "headline_auc": round(float(rcv.mean()), 3), "headline_auc_std": round(float(rcv.std()), 3),
    "test_auc": round(float(roc_auc_score(yte, prob)), 4),
    "test_avg_precision": round(float(average_precision_score(yte, prob)), 4),
    "at_0.5": at(0.5), "at_chosen_threshold": at(threshold),
    "feature_importance": importance, "segments": segments,
}
print(json.dumps({k: metrics[k] for k in ["test_auc", "at_0.5", "at_chosen_threshold"]}, indent=2))

joblib.dump({"model": model, "threshold": threshold, "features": FEATURES}, "model.joblib")
json.dump(metrics, open("metrics.json", "w"), indent=2)
print("saved model.joblib and metrics.json")
