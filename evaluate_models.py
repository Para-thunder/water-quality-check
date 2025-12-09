import numpy as np
import pandas as pd
from pathlib import Path

# Sklearn imports
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
    recall_score, f1_score, roc_auc_score
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
# Commonly: 1 = potable, 0 = not potable. If you want to detect unsafe samples, set POSITIVE_LABEL = 0
POSITIVE_LABEL = 1

# Load dataset
df = pd.read_csv(DATA_PATH)
print("Columns:", df.columns.tolist())

# Update this if your target column name is different
target_col = "Potability"
if target_col not in df.columns:
    raise ValueError(f"Target column '{target_col}' not found. Update target_col variable.")

X = df.drop(columns=[target_col])
y = df[target_col]

# Train/test split (stratify to preserve class balance)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

# Basic imputation and pipelines
imputer = SimpleImputer(strategy="median")

models = {
    "Logistic Regression": make_pipeline(imputer, StandardScaler(), LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)),
    "Decision Tree": make_pipeline(imputer, DecisionTreeClassifier(random_state=RANDOM_STATE)),
    "Random Forest": make_pipeline(imputer, RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)),
    "SVM": make_pipeline(imputer, StandardScaler(), SVC(probability=True, random_state=RANDOM_STATE))
}

if has_xgb:
    models["XGBoost"] = make_pipeline(imputer, XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=RANDOM_STATE))

rows = []
for name, clf in models.items():
    print(f"Training {name} ...")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Get predicted probability for POSITIVE_LABEL if possible
    y_prob = None
    try:
        probs = clf.predict_proba(X_test)
        # get classes from final estimator if pipeline
        final = list(clf.named_steps.values())[-1]
        classes = getattr(final, "classes_", None)
        if classes is not None and POSITIVE_LABEL in classes:
            pos_index = list(classes).index(POSITIVE_LABEL)
            y_prob = probs[:, pos_index]
        else:
            # fallback to column 1 if binary
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

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label=POSITIVE_LABEL, zero_division=0)
    rec = recall_score(y_test, y_pred, pos_label=POSITIVE_LABEL, zero_division=0)
    f1 = f1_score(y_test, y_pred, pos_label=POSITIVE_LABEL, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob) if y_prob is not None else np.nan

    rows.append({
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-score": f1,
        "ROC-AUC": roc_auc
    })

metrics_df = pd.DataFrame(rows).set_index("Model")
metrics_df = metrics_df[["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]]
metrics_df_display = metrics_df.round(4)
print("\nSummary metrics:")
print(metrics_df_display)

metrics_df.to_csv("model_metrics_summary.csv")
print("\nSaved summary to model_metrics_summary.csv")