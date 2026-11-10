# =============================================================
# FILE: src/03_train.py
# PURPOSE: Professional model training pipeline
# RUN: python src/03_train.py
# =============================================================

import pandas as pd
import numpy as np
import warnings
import joblib
import optuna

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# -------------------------------------------------------------
# 1. LOAD DATA
# -------------------------------------------------------------
print("=" * 60)
print("STEP 1: LOADING DATA")
print("=" * 60)

X_train = pd.read_csv("data/processed/X_train.csv")
y_train = pd.read_csv("data/processed/y_train.csv").squeeze()

X_val = pd.read_csv("data/processed/X_val.csv")
y_val = pd.read_csv("data/processed/y_val.csv").squeeze()

X_test = pd.read_csv("data/processed/X_test.csv")
y_test = pd.read_csv("data/processed/y_test.csv").squeeze()

X_train_smote = pd.read_csv("data/processed/X_train_smote.csv")
y_train_smote = pd.read_csv("data/processed/y_train_smote.csv").squeeze()

X_train_under = pd.read_csv("data/processed/X_train_under.csv")
y_train_under = pd.read_csv("data/processed/y_train_under.csv").squeeze()

scale_pos_weight = joblib.load(
    "artifacts/scale_pos_weight.pkl"
)

print("\nDatasets Loaded Successfully")

# -------------------------------------------------------------
# 2. HELPER FUNCTION
# -------------------------------------------------------------
def evaluate_model(
    model_name,
    model,
    X_eval,
    y_eval,
    threshold=0.5
):
    """
    Evaluate model performance.
    """

    probs = model.predict_proba(X_eval)[:,1]

    preds = (probs >= threshold).astype(int)

    f1 = f1_score(y_eval, preds)

    roc_auc = roc_auc_score(y_eval, probs)

    pr_auc = average_precision_score(y_eval, probs)

    tn, fp, fn, tp = confusion_matrix(
        y_eval,
        preds
    ).ravel()

    print("\n" + "─" * 50)
    print(f"MODEL: {model_name}")
    print("─" * 50)

    print(f"F1 Score   : {f1:.4f}")
    print(f"ROC-AUC    : {roc_auc:.4f}")
    print(f"PR-AUC     : {pr_auc:.4f}")

    print("\nConfusion Matrix:")

    print(f"True Positives  : {tp}")
    print(f"False Negatives : {fn}")
    print(f"False Positives : {fp}")
    print(f"True Negatives  : {tn}")

    return {
        "model": model_name,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "tp": tp,
        "fn": fn,
        "fp": fp
    }

results = []

# -------------------------------------------------------------
# 3. LOGISTIC REGRESSION BASELINE
# -------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 2: LOGISTIC REGRESSION")
print("=" * 60)

lr_model = LogisticRegression(
    max_iter=1000,
    class_weight='balanced',
    random_state=42
)

print("\nTraining Logistic Regression...")

lr_model.fit(
    X_train_smote,
    y_train_smote
)

result = evaluate_model(
    "Logistic Regression",
    lr_model,
    X_val,
    y_val
)

results.append(result)

joblib.dump(
    lr_model,
    "artifacts/model_logistic.pkl"
)

print("\nSaved: model_logistic.pkl")

# -------------------------------------------------------------
# 4. RANDOM FOREST
# -------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 3: RANDOM FOREST")
print("=" * 60)

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    n_jobs=-1,
    class_weight='balanced',
    random_state=42
)

print("\nTraining Random Forest...")

rf_model.fit(
    X_train_smote,
    y_train_smote
)

result = evaluate_model(
    "Random Forest",
    rf_model,
    X_val,
    y_val
)

results.append(result)

joblib.dump(
    rf_model,
    "artifacts/model_rf.pkl"
)

print("\nSaved: model_rf.pkl")

# -------------------------------------------------------------
# 5. XGBOOST + SMOTE
# -------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 4: XGBOOST + SMOTE")
print("=" * 60)

xgb_smote = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='aucpr',
    random_state=42,
    verbosity=0
)

print("\nTraining XGBoost + SMOTE...")

xgb_smote.fit(
    X_train_smote,
    y_train_smote
)

result = evaluate_model(
    "XGBoost + SMOTE",
    xgb_smote,
    X_val,
    y_val
)

results.append(result)

joblib.dump(
    xgb_smote,
    "artifacts/model_xgb_smote.pkl"
)

print("\nSaved: model_xgb_smote.pkl")

# -------------------------------------------------------------
# 6. XGBOOST COST-SENSITIVE
# -------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 5: COST-SENSITIVE XGBOOST")
print("=" * 60)

print(f"\nscale_pos_weight = {scale_pos_weight:.2f}")

xgb_cost = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=scale_pos_weight,
    eval_metric='aucpr',
    random_state=42,
    verbosity=0
)

print("\nTraining Cost-Sensitive XGBoost...")

xgb_cost.fit(
    X_train,
    y_train
)

result = evaluate_model(
    "XGBoost + Cost Sensitive",
    xgb_cost,
    X_val,
    y_val
)

results.append(result)

joblib.dump(
    xgb_cost,
    "artifacts/model_xgb_cost.pkl"
)

print("\nSaved: model_xgb_cost.pkl")

# -------------------------------------------------------------
# 7. OPTUNA HYPERPARAMETER TUNING
# -------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 6: OPTUNA HYPERPARAMETER TUNING")
print("=" * 60)

print("""
Optuna will:
- try multiple hyperparameter combinations
- optimize PR-AUC
- use validation set correctly
""")

def objective(trial):

    params = {

        'n_estimators': trial.suggest_int(
            'n_estimators',
            100,
            500
        ),

        'max_depth': trial.suggest_int(
            'max_depth',
            3,
            10
        ),

        'learning_rate': trial.suggest_float(
            'learning_rate',
            0.01,
            0.3,
            log=True
        ),

        'subsample': trial.suggest_float(
            'subsample',
            0.5,
            1.0
        ),

        'colsample_bytree': trial.suggest_float(
            'colsample_bytree',
            0.5,
            1.0
        ),

        'min_child_weight': trial.suggest_int(
            'min_child_weight',
            1,
            10
        ),

        'gamma': trial.suggest_float(
            'gamma',
            0,
            5
        ),

        'reg_alpha': trial.suggest_float(
            'reg_alpha',
            0,
            2
        ),

        'reg_lambda': trial.suggest_float(
            'reg_lambda',
            0,
            2
        )
    }

    model = XGBClassifier(
        **params,
        eval_metric='aucpr',
        random_state=42,
        verbosity=0
    )

    model.fit(
        X_train_smote,
        y_train_smote
    )

    probs = model.predict_proba(X_val)[:,1]

    pr_auc = average_precision_score(
        y_val,
        probs
    )

    return pr_auc

print("\nStarting Optuna optimization...")

study = optuna.create_study(
    direction='maximize'
)

study.optimize(
    objective,
    n_trials=30
)

print("\nBest PR-AUC:")
print(study.best_value)

print("\nBest Parameters:")
print(study.best_params)

# -------------------------------------------------------------
# 8. TRAIN FINAL BEST MODEL
# -------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 7: FINAL BEST MODEL")
print("=" * 60)

best_model = XGBClassifier(
    **study.best_params,
    eval_metric='aucpr',
    random_state=42,
    verbosity=0
)

print("\nTraining Final Best Model...")

best_model.fit(
    X_train_smote,
    y_train_smote
)

result = evaluate_model(
    "XGBoost + Optuna",
    best_model,
    X_val,
    y_val
)

results.append(result)

joblib.dump(
    best_model,
    "artifacts/model_best.pkl"
)

joblib.dump(
    study.best_params,
    "artifacts/best_params.pkl"
)

print("\nSaved:")
print("- model_best.pkl")
print("- best_params.pkl")

# -------------------------------------------------------------
# 9. FINAL MODEL COMPARISON
# -------------------------------------------------------------
print("\n" + "=" * 60)
print("FINAL MODEL COMPARISON")
print("=" * 60)

results_df = pd.DataFrame(results)

print("\n")
print(results_df[
    [
        'model',
        'f1',
        'roc_auc',
        'pr_auc',
        'tp',
        'fn',
        'fp'
    ]
])

# -------------------------------------------------------------
# 10. SELECT BEST MODEL
# -------------------------------------------------------------
best_result = results_df.sort_values(
    by='pr_auc',
    ascending=False
).iloc[0]

print("\n" + "=" * 60)
print("BEST MODEL")
print("=" * 60)

print(f"""
Model        : {best_result['model']}
PR-AUC       : {best_result['pr_auc']:.4f}
F1 Score     : {best_result['f1']:.4f}
""")

print("""
PR-AUC is prioritized because:
- dataset is highly imbalanced
- fraud detection focuses on positive class
- PR-AUC better reflects fraud-catching ability
""")

print("\nTraining Completed Successfully!")

print("\nNext Step: Evaluation + SHAP")