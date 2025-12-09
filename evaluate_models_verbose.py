import numpy as np
import pandas as pd
from pathlib import Path
import json

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, average_precision_score
)

# Optional: XGBoost (install with `pip install xgboost`)
try:
    from xgboost import XGBClassifier
    has_xgb = True
except Exception:
    has_xgb = False

# CONFIG
DATA_PATH = Path("data/raw/water_potability.csv")
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Set this to the label you want to treat as "positive" for metrics.
# If you want to detect unsafe/non-potable, set POSITIVE_LABEL = 0
POSITIVE_LABEL = 1

# Toggle to use class weights ('balanced') for applicable classifiers
USE_CLASS_WEIGHT_BALANCED = True

# Load dataset
df = pd.read_csv(DATA_PATH)
print("Columns:", df.columns.tolist())

target_col = "Potability"  # update if different
if target_col not in df.columns:
    raise ValueError(f"Target column '{target_col}' not found. Update target_col variable.")

X = df.drop(columns=[target_col])
y = df[target_col]

# Show class distribution
class_counts = y.value_counts().sort_index()
print("\nClass distribution (label: count):")
for lbl, cnt in class_counts.items():
    print(f"  {lbl}: {cnt}")
print(f"Total samples: {len(y)}")

# Train/test split (stratify)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

imputer = SimpleImputer(strategy="median")

# Helper to build final estimator with optional class weight
def final_estimator(name):
    if name == "Logistic Regression":
        if USE_CLASS_WEIGHT_BALANCED:
            return LogisticRegression(random_state=RANDOM_STATE, max_iter=1000, class_weight='balanced')
        else:
            return LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
    if name == "Decision Tree":
        return DecisionTreeClassifier(random_state=RANDOM_STATE, class_weight='balanced' if USE_CLASS_WEIGHT_BALANCED else None)
    if name == "Random Forest":
        return RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, class_weight='balanced' if USE_CLASS_WEIGHT_BALANCED else None)
    if name == "SVM":
        return SVC(probability=True, random_state=RANDOM_STATE, class_weight='balanced' if USE_CLASS_WEIGHT_BALANCED else None)
    if name == "XGBoost":
        return XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=RANDOM_STATE)

# Models dictionary (pipelines)
models = {}
for mname in ["Logistic Regression", "Decision Tree", "Random Forest", "SVM"]:
    est = final_estimator(mname)
    if mname in ["Logistic Regression", "SVM"]:
        pipe = make_pipeline(imputer, StandardScaler(), est)
    else:
        pipe = make_pipeline(imputer, est)
    models[mname] = pipe

if has_xgb:
    models["XGBoost"] = make_pipeline(imputer, final_estimator("XGBoost"))

results = []

for name, clf in models.items():
    print(f"\n--- Training {name} ---")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    # try to get probability for POSITIVE_LABEL
    y_prob = None
    try:
        probs = clf.predict_proba(X_test)
        final = list(clf.named_steps.values())[-1]
        classes = getattr(final, "classes_", None)
        if classes is not None and POSITIVE_LABEL in classes:
            pos_idx = list(classes).index(POSITIVE_LABEL)
            y_prob = probs[:, pos_idx]
        else:
            y_prob = probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]
    except Exception:
        try:
            dec = clf.decision_function(X_test)
            from scipy.special import expit
            y_prob = expit(dec)
            if y_prob.ndim > 1:
                y_prob = y_prob[:, 1] if y_prob.shape[1] > 1 else y_prob[:, 0]
        except Exception:
            y_prob = None

    cm = confusion_matrix(y_test, y_pred, labels=[0,1])
    # Ensure native Python ints for JSON serialization
    cm_list = [[int(x) for x in row] for row in cm.tolist()]

    # Show predicted-class distribution (native ints)
    unique, counts = np.unique(y_pred, return_counts=True)
    pred_dist = {int(k): int(v) for k, v in zip(unique, counts)}
    print("Confusion matrix (rows=true [0,1], cols=pred [0,1]):")
    print(cm)
    print("Predicted class counts:", pred_dist)

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, pos_label=POSITIVE_LABEL, zero_division=0))
    rec = float(recall_score(y_test, y_pred, pos_label=POSITIVE_LABEL, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, pos_label=POSITIVE_LABEL, zero_division=0))
    roc_auc = float(roc_auc_score(y_test, y_prob)) if y_prob is not None else float("nan")
    pr_auc = float(average_precision_score(y_test, y_prob)) if y_prob is not None else float("nan")

    print(f"Metrics for POSITIVE_LABEL={POSITIVE_LABEL}: Accuracy={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}, ROC-AUC={roc_auc:.4f}, PR-AUC={pr_auc:.4f}")

    results.append({
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-score": f1,
        "ROC-AUC": roc_auc,
        "PR-AUC": pr_auc,
        "Predicted_counts": pred_dist,
        "Confusion_matrix": cm_list
    })

# Save summary
metrics_df = pd.DataFrame(results).set_index("Model")
metrics_df = metrics_df[["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC", "PR-AUC"]]
metrics_df.to_csv("model_metrics_verbose_summary.csv")
print("\nSaved summary to model_metrics_verbose_summary.csv")

# Also save raw results json-like for debugging (now JSON-serializable)
with open("model_metrics_verbose_details.json", "w") as f:
    json.dump(results, f, indent=2)
print("Saved detailed results to model_metrics_verbose_details.json")