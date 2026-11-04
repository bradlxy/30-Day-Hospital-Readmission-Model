import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve
)
from xgboost import XGBClassifier

RANDOM_STATE = 42

# ============================================================
# 1. Load data
# ============================================================
print("Loading modeling-ready dataset...")
df = pd.read_csv("mimiciv_readmission_ready.csv")
print(f"Loaded shape: {df.shape}")

target_col = "target_30day_readmit"
print(f"Class balance:\n{df[target_col].value_counts()}")
print(f"Positive rate: {df[target_col].mean():.3%}")

if df[target_col].sum() < 10:
    print("\n*** WARNING: fewer than 10 positive cases. Model results below "
          "will not be statistically meaningful - this is expected with the "
          "MIMIC-IV demo (~100 patients) and is mainly useful for validating "
          "the pipeline runs end-to-end, not for judging real performance. ***\n")

# ============================================================
# 2. Feature prep
# ============================================================
# Drop identifier columns that shouldn't be model inputs
id_cols = ["subject_id", "hadm_id"]
X = df.drop(columns=id_cols + [target_col])
y = df[target_col]

# Encode categorical columns (gender, care_unit_name) as one-hot
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
print(f"Encoding categorical columns: {categorical_cols}")
X = pd.get_dummies(X, columns=categorical_cols, dummy_na=True)

# Fill any remaining missing values (comorbidity flags should be 0/1 already)
X = X.fillna(0)

print(f"Final feature matrix shape: {X.shape}")

# ============================================================
# 3. Train/test split (stratified to preserve rare-class ratio)
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"Train positives: {y_train.sum()}, Test positives: {y_test.sum()}")

# Standardize features for logistic regression (tree models don't need this,
# but it doesn't hurt them either since we're using separate pipelines)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================
# 4. Train models
# ============================================================
# class imbalance handling: class_weight for LR/RF, scale_pos_weight for XGBoost
n_pos = y_train.sum()
n_neg = len(y_train) - n_pos
scale_pos_weight = (n_neg / n_pos) if n_pos > 0 else 1

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, class_weight="balanced", random_state=RANDOM_STATE
    ),
    "XGBoost": XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        scale_pos_weight=scale_pos_weight, eval_metric="logloss",
        random_state=RANDOM_STATE
    ),
}

results = {}

print("\n" + "=" * 60)
print("TRAINING MODELS")
print("=" * 60)

for name, model in models.items():
    print(f"\n--- {name} ---")

    if name == "Logistic Regression":
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

    # Metrics - ROC-AUC and precision/recall matter more than accuracy here
    # given class imbalance
    try:
        auc = roc_auc_score(y_test, y_proba)
    except ValueError:
        auc = float("nan")  # happens if y_test has only one class present

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    print(f"ROC-AUC:   {auc:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1:        {f1:.3f}")
    print(f"Confusion matrix:\n{cm}")
    print(classification_report(y_test, y_pred, zero_division=0))

    results[name] = {
        "model": model,
        "auc": auc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

# ============================================================
# 5. Comparison summary -> CSV for R
# ============================================================
print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)
summary_df = pd.DataFrame({
    name: {k: v for k, v in metrics.items() if k != "model"}
    for name, metrics in results.items()
}).T
summary_df.index.name = "model"
print(summary_df.round(3))
summary_df.to_csv("model_comparison_summary.csv")
print("Saved: model_comparison_summary.csv")

# ============================================================
# 6. Feature importance (tree-based models) -> CSV for R
# ============================================================
importance_frames = []
for name in ["Random Forest", "XGBoost"]:
    model = results[name]["model"]
    importances = pd.Series(model.feature_importances_, index=X.columns, name="importance")
    imp_df = importances.reset_index()
    imp_df.columns = ["feature", "importance"]
    imp_df["model"] = name
    importance_frames.append(imp_df)

feature_importance_df = pd.concat(importance_frames, ignore_index=True)
feature_importance_df.to_csv("feature_importance.csv", index=False)
print("Saved: feature_importance.csv")

# ============================================================
# 7. Logistic regression coefficients -> CSV for R
# (direction of effect, not just magnitude - positive coef = higher readmit odds)
# ============================================================
lr_model = results["Logistic Regression"]["model"]
lr_coefs = pd.DataFrame({
    "feature": X.columns,
    "coefficient": lr_model.coef_[0]
}).sort_values("coefficient", key=abs, ascending=False)
lr_coefs.to_csv("logistic_regression_coefficients.csv", index=False)
print("Saved: logistic_regression_coefficients.csv")

# ============================================================
# 8. Test-set predictions + probabilities -> CSV for R
# (for ROC curves, confusion matrices, calibration plots, etc. in R)
# ============================================================
pred_frames = []
for name, model in models.items():
    if name == "Logistic Regression":
        proba = model.predict_proba(X_test_scaled)[:, 1]
        pred = model.predict(X_test_scaled)
    else:
        proba = model.predict_proba(X_test)[:, 1]
        pred = model.predict(X_test)

    pred_df = pd.DataFrame({
        "row_id": X_test.index,
        "y_true": y_test.values,
        "y_pred": pred,
        "y_proba": proba,
        "model": name
    })
    pred_frames.append(pred_df)

predictions_df = pd.concat(pred_frames, ignore_index=True)
predictions_df.to_csv("test_predictions.csv", index=False)
print("Saved: test_predictions.csv")

print("\nDone. All outputs saved as CSV for downstream plotting in R:")
print("  - model_comparison_summary.csv")
print("  - feature_importance.csv")
print("  - logistic_regression_coefficients.csv")
print("  - test_predictions.csv")