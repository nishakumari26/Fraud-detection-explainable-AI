# =============================================================
# FILE: src/05_threshold.py
# PURPOSE: Business threshold optimization
# RUN: python src/05_threshold.py
# =============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
import warnings

from sklearn.metrics import (
    confusion_matrix,
    f1_score
)

warnings.filterwarnings("ignore")

# -------------------------------------------------------------
# CREATE PLOTS DIRECTORY
# -------------------------------------------------------------
os.makedirs("plots", exist_ok=True)

# -------------------------------------------------------------
# 1. LOAD DATA & MODEL
# -------------------------------------------------------------
print("=" * 60)
print("STEP 1: LOADING DATA & MODEL")
print("=" * 60)

X_val = pd.read_csv(
    "data/processed/X_val.csv"
)

y_val = pd.read_csv(
    "data/processed/y_val.csv"
).squeeze()

model = joblib.load(
    "artifacts/model_best.pkl"
)

print("\nValidation data loaded")

# -------------------------------------------------------------
# 2. PREDICT PROBABILITIES
# -------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 2: GENERATING PROBABILITIES")
print("=" * 60)

probs = model.predict_proba(
    X_val
)[:,1]

print(f"\nGenerated {len(probs):,} probabilities")

# -------------------------------------------------------------
# 3. BUSINESS COST ASSUMPTIONS
# -------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 3: BUSINESS COST MODEL")
print("=" * 60)

FN_COST = 200
FP_COST = 10

print(f"""
Business Assumptions:

False Negative Cost : ${FN_COST}
False Positive Cost : ${FP_COST}

Meaning:
Missing fraud is {FN_COST/FP_COST:.0f}x
more expensive than false alarms.
""")

# -------------------------------------------------------------
# 4. THRESHOLD SCANNING
# -------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 4: THRESHOLD SCANNING")
print("=" * 60)

thresholds = np.arange(
    0.01,
    0.99,
    0.01
)

records = []

for threshold in thresholds:

    preds = (
        probs >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_val,
        preds
    ).ravel()

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0 else 0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0 else 0
    )

    f1 = f1_score(
        y_val,
        preds,
        zero_division=0
    )

    total_cost = (
        fn * FN_COST
    ) + (
        fp * FP_COST
    )

    records.append({
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "tn": tn,
        "total_cost": total_cost
    })

results = pd.DataFrame(records)

# -------------------------------------------------------------
# 5. FIND BEST THRESHOLD
# -------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 5: OPTIMAL THRESHOLD")
print("=" * 60)

best_cost_row = results.loc[
    results['total_cost'].idxmin()
]

best_f1_row = results.loc[
    results['f1'].idxmax()
]

default_row = results[
    results['threshold'] == 0.50
].iloc[0]

print("\nBEST COST-OPTIMAL THRESHOLD")

print(f"""
Threshold   : {best_cost_row['threshold']:.2f}

Precision   : {best_cost_row['precision']:.4f}
Recall      : {best_cost_row['recall']:.4f}
F1 Score    : {best_cost_row['f1']:.4f}

False Positives : {int(best_cost_row['fp'])}
False Negatives : {int(best_cost_row['fn'])}

Total Cost : ${best_cost_row['total_cost']:,.0f}
""")

print("\nBEST F1 THRESHOLD")

print(f"""
Threshold   : {best_f1_row['threshold']:.2f}
F1 Score    : {best_f1_row['f1']:.4f}
Total Cost  : ${best_f1_row['total_cost']:,.0f}
""")

print("\nDEFAULT THRESHOLD (0.5)")

print(f"""
F1 Score   : {default_row['f1']:.4f}
Total Cost : ${default_row['total_cost']:,.0f}
""")

# -------------------------------------------------------------
# 6. VISUALIZATION
# -------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 6: VISUALIZATION")
print("=" * 60)

fig, axes = plt.subplots(
    2,
    2,
    figsize=(15,10)
)

# ---------------- COST CURVE ----------------
axes[0,0].plot(
    results['threshold'],
    results['total_cost'],
    linewidth=2
)

axes[0,0].axvline(
    best_cost_row['threshold'],
    linestyle='--',
    label=f"Optimal = {best_cost_row['threshold']:.2f}"
)

axes[0,0].set_title(
    "Business Cost vs Threshold"
)

axes[0,0].set_xlabel("Threshold")
axes[0,0].set_ylabel("Total Cost")

axes[0,0].legend()

# ---------------- F1 CURVE ----------------
axes[0,1].plot(
    results['threshold'],
    results['f1'],
    linewidth=2
)

axes[0,1].axvline(
    best_f1_row['threshold'],
    linestyle='--',
    label=f"Best F1 = {best_f1_row['threshold']:.2f}"
)

axes[0,1].set_title(
    "F1 Score vs Threshold"
)

axes[0,1].set_xlabel("Threshold")
axes[0,1].set_ylabel("F1 Score")

axes[0,1].legend()

# ---------------- PRECISION VS RECALL ----------------
axes[1,0].plot(
    results['threshold'],
    results['precision'],
    label='Precision'
)

axes[1,0].plot(
    results['threshold'],
    results['recall'],
    label='Recall'
)

axes[1,0].set_title(
    "Precision vs Recall"
)

axes[1,0].set_xlabel("Threshold")

axes[1,0].legend()

# ---------------- FP vs FN ----------------
axes[1,1].plot(
    results['threshold'],
    results['fp'],
    label='False Positives'
)

axes[1,1].plot(
    results['threshold'],
    results['fn'],
    label='False Negatives'
)

axes[1,1].set_title(
    "False Positives vs False Negatives"
)

axes[1,1].set_xlabel("Threshold")

axes[1,1].legend()

plt.tight_layout()

plt.savefig(
    "plots/13_threshold_analysis.png",
    dpi=300
)

plt.close()

print("\nSaved: plots/13_threshold_analysis.png")

# -------------------------------------------------------------
# 7. SAVE OPTIMAL THRESHOLD
# -------------------------------------------------------------
optimal_threshold = float(
    best_cost_row['threshold']
)

joblib.dump(
    optimal_threshold,
    "artifacts/optimal_threshold.pkl"
)

results.to_csv(
    "artifacts/threshold_analysis.csv",
    index=False
)

print("\nSaved:")
print("- optimal_threshold.pkl")
print("- threshold_analysis.csv")

# -------------------------------------------------------------
# 8. FINAL BUSINESS INSIGHTS
# -------------------------------------------------------------
print("\n" + "=" * 60)
print("FINAL BUSINESS INSIGHTS")
print("=" * 60)

print(f"""
1. Default threshold 0.5 was NOT optimal

2. Optimal threshold found:
   {optimal_threshold:.2f}

3. Threshold optimization reduced
   operational fraud cost

4. Fraud systems should optimize
   BUSINESS VALUE, not only ML metrics

5. False negatives are substantially
   more expensive than false positives
""")

print("\nThreshold Optimization Completed!")