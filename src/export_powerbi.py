# =============================================================
# FILE: src/export_powerbi.py
# PURPOSE: Export predictions for Power BI dashboards
# =============================================================
#
# WHAT IS THIS FILE?
# ------------------
# This script runs the trained model on the test dataset and
# exports the results as a CSV file that can be imported into
# Power BI for business dashboarding.
#
# WHAT IS POWER BI?
# -----------------
# Power BI is Microsoft's data visualization tool (like Tableau).
# Business analysts use it to create interactive dashboards.
# By exporting our data in CSV format, we make it easy for
# non-technical people to analyze fraud patterns.
#
# WHY IS THIS IMPORTANT FOR INTERVIEWS?
# --------------------------------------
# It shows you can bridge the gap between data science and
# business — you don't just build models, you deliver insights
# that business people can actually use.
#
# HOW TO RUN:
# python src/export_powerbi.py
# =============================================================

# ---------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------

import pandas as pd
import numpy as np
import joblib
import os
import shap
import warnings
from datetime import datetime, timedelta
import random

warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

# Find the project root directory
BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

# Create the powerbi output directory
POWERBI_DIR = os.path.join(BASE_DIR, "powerbi")
os.makedirs(POWERBI_DIR, exist_ok=True)

# ---------------------------------------------------------
# 1. LOAD MODEL AND DATA
# ---------------------------------------------------------
print("=" * 60)
print("STEP 1: LOADING MODEL AND DATA")
print("=" * 60)

# Load the trained model
model = joblib.load(
    os.path.join(BASE_DIR, "artifacts/model_best.pkl")
)

# Load the optimal threshold
threshold = joblib.load(
    os.path.join(BASE_DIR, "artifacts/optimal_threshold.pkl")
)

# Load test data
X_test = pd.read_csv(
    os.path.join(BASE_DIR, "data/processed/X_test.csv")
)

y_test = pd.read_csv(
    os.path.join(BASE_DIR, "data/processed/y_test.csv")
).squeeze()

# Load feature names
feature_names = joblib.load(
    os.path.join(BASE_DIR, "artifacts/feature_names.pkl")
)

print(f"\nModel loaded successfully")
print(f"Test data shape: {X_test.shape}")
print(f"Threshold: {threshold}")

# ---------------------------------------------------------
# 2. GENERATE PREDICTIONS
# ---------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 2: GENERATING PREDICTIONS")
print("=" * 60)

# Get fraud probabilities for all test transactions
# predict_proba returns [P(legit), P(fraud)] for each row
# [:,1] gets just P(fraud)
probs = model.predict_proba(X_test)[:, 1]

# Apply threshold to get binary predictions
predictions = (probs >= threshold).astype(int)

print(f"\nTotal predictions: {len(predictions):,}")
print(f"Fraud detected: {predictions.sum():,}")
print(f"Legitimate: {(1-predictions).sum():,}")

# ---------------------------------------------------------
# 3. ASSIGN RISK LEVELS
# ---------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 3: ASSIGNING RISK LEVELS")
print("=" * 60)

# Create risk level for each transaction
def get_risk_level(prob):
    """
    Categorize fraud probability into risk levels.

    This is how real fraud systems work — they don't just
    say "fraud" or "not fraud". They assign risk levels
    so analysts can prioritize their work.
    """
    if prob >= 0.8:
        return "Critical"
    elif prob >= 0.5:
        return "High"
    elif prob >= threshold:
        return "Medium"
    else:
        return "Low"

# Apply to all probabilities
risk_levels = [get_risk_level(p) for p in probs]

# Count by risk level
risk_counts = pd.Series(risk_levels).value_counts()
print("\nRisk Level Distribution:")
for level, count in risk_counts.items():
    print(f"  {level:10s}: {count:,}")

# ---------------------------------------------------------
# 4. CALCULATE TOP SHAP FEATURES (for a sample)
# ---------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 4: CALCULATING SHAP FEATURES")
print("=" * 60)

print("\nComputing SHAP values for all test transactions...")
print("(This may take a minute...)")

# Create SHAP explainer
explainer = shap.TreeExplainer(model)

# Calculate SHAP values for all test data
shap_values = explainer.shap_values(X_test)

# For each transaction, find the top contributing feature
top_feature_names = []
top_feature_values = []

for i in range(len(X_test)):

    # Get SHAP values for this transaction
    single_shap = shap_values[i]

    # Find the feature with the highest absolute SHAP value
    top_idx = np.argmax(np.abs(single_shap))

    top_feature_names.append(feature_names[top_idx])

    top_feature_values.append(
        round(float(single_shap[top_idx]), 4)
    )

print(f"\nSHAP analysis complete for {len(X_test):,} transactions")

# ---------------------------------------------------------
# 5. CREATE SIMULATED TIMESTAMPS
# ---------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 5: CREATING REALISTIC TIMESTAMPS")
print("=" * 60)

# The original dataset doesn't have real dates, so we
# create realistic-looking timestamps for Power BI
# This makes the dashboard look more professional

# Start from 30 days ago
base_date = datetime.now() - timedelta(days=30)

timestamps = []

for i in range(len(X_test)):

    # Spread transactions evenly over 30 days
    hours_offset = (i / len(X_test)) * 30 * 24

    # Add some randomness (±2 hours)
    hours_offset += random.uniform(-2, 2)

    ts = base_date + timedelta(hours=hours_offset)

    timestamps.append(ts.strftime("%Y-%m-%d %H:%M:%S"))

print(f"\nGenerated {len(timestamps):,} timestamps")
print(f"Date range: {timestamps[0]} to {timestamps[-1]}")

# ---------------------------------------------------------
# 6. BUILD THE EXPORT DATAFRAME
# ---------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 6: BUILDING EXPORT DATA")
print("=" * 60)

# Create the main DataFrame for Power BI
export_df = pd.DataFrame({

    # Transaction identifier
    "transaction_id": range(1, len(X_test) + 1),

    # Timestamp for time-series analysis in Power BI
    "timestamp": timestamps,

    # Extract date and hour for easy filtering
    "date": [t[:10] for t in timestamps],
    "hour_of_day": X_test["Hour"].round(0).astype(int).values
        if "Hour" in X_test.columns else 0,

    # Transaction amount (reverse the log transform)
    # np.expm1 is the inverse of np.log1p
    "amount_usd": np.round(
        np.expm1(X_test["Amount_log"].values), 2
    ) if "Amount_log" in X_test.columns else 0,

    # Model predictions
    "fraud_probability": np.round(probs, 4),
    "is_fraud_predicted": predictions,
    "actual_label": y_test.values,
    "risk_level": risk_levels,

    # Prediction correctness
    "is_correct": (predictions == y_test.values).astype(int),

    # SHAP explainability
    "top_shap_feature": top_feature_names,
    "top_shap_value": top_feature_values,
})

# Add a "prediction_type" column for Power BI filtering
# This creates labels like: True Positive, False Negative, etc.
def get_prediction_type(row):
    """
    Classify each prediction into one of 4 categories:
    - True Positive: Correctly caught fraud
    - True Negative: Correctly approved legitimate
    - False Positive: Wrongly flagged legitimate as fraud
    - False Negative: Missed actual fraud (WORST case)
    """
    if row["actual_label"] == 1 and row["is_fraud_predicted"] == 1:
        return "True Positive"
    elif row["actual_label"] == 0 and row["is_fraud_predicted"] == 0:
        return "True Negative"
    elif row["actual_label"] == 0 and row["is_fraud_predicted"] == 1:
        return "False Positive"
    else:
        return "False Negative"

export_df["prediction_type"] = export_df.apply(
    get_prediction_type, axis=1
)

print(f"\nExport DataFrame shape: {export_df.shape}")
print(f"\nColumns:")
for col in export_df.columns:
    print(f"  - {col}")

# ---------------------------------------------------------
# 7. SAVE TO CSV
# ---------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 7: SAVING FILES")
print("=" * 60)

# Main export file
output_path = os.path.join(
    POWERBI_DIR,
    "fraud_dashboard_data.csv"
)

export_df.to_csv(output_path, index=False)

print(f"\n[OK] Main data exported to:")
print(f"   {output_path}")

# Also create a summary table for Power BI cards/KPIs
summary = {
    "metric": [
        "Total Transactions",
        "Fraud Detected",
        "Legitimate",
        "Fraud Rate (%)",
        "Model Accuracy (%)",
        "Critical Risk Count",
        "High Risk Count",
        "Medium Risk Count",
        "Low Risk Count",
        "False Negatives (Missed Fraud)",
        "False Positives (False Alarms)",
        "Threshold Used"
    ],
    "value": [
        len(export_df),
        int(predictions.sum()),
        int((1 - predictions).sum()),
        round(predictions.mean() * 100, 2),
        round(
            (predictions == y_test.values).mean() * 100, 2
        ),
        int(risk_counts.get("Critical", 0)),
        int(risk_counts.get("High", 0)),
        int(risk_counts.get("Medium", 0)),
        int(risk_counts.get("Low", 0)),
        int(export_df[
            export_df["prediction_type"] == "False Negative"
        ].shape[0]),
        int(export_df[
            export_df["prediction_type"] == "False Positive"
        ].shape[0]),
        round(float(threshold), 4)
    ]
}

summary_df = pd.DataFrame(summary)

summary_path = os.path.join(
    POWERBI_DIR,
    "fraud_summary_kpis.csv"
)

summary_df.to_csv(summary_path, index=False)

print(f"\n[OK] Summary KPIs exported to:")
print(f"   {summary_path}")

# ---------------------------------------------------------
# 8. PRINT SUMMARY
# ---------------------------------------------------------
print("\n" + "=" * 60)
print("EXPORT COMPLETE!")
print("=" * 60)

print(f"""
Files created in: {POWERBI_DIR}

1. fraud_dashboard_data.csv
   - {len(export_df):,} rows x {len(export_df.columns)} columns
   - Every test transaction with predictions + SHAP

2. fraud_summary_kpis.csv
   - Key performance indicators for dashboard cards

HOW TO USE IN POWER BI:
1. Open Power BI Desktop
2. Click "Get Data" -> "Text/CSV"
3. Select fraud_dashboard_data.csv
4. Create visualizations:
   - Fraud Rate by Hour (bar chart)
   - Risk Level Distribution (donut chart)
   - Predictions Over Time (line chart)
   - Top SHAP Features (tree map)
   - KPI Cards from summary file

SUGGESTED POWER BI VISUALS:
- Card: Total Transactions, Fraud Rate
- Donut Chart: Risk Level Distribution
- Bar Chart: Fraud by Hour of Day
- Scatter Plot: Amount vs Fraud Probability
- Table: Top False Negatives (missed frauds)
- Line Chart: Daily Fraud Trend
""")

# ---------------------------------------------------------
# 9. PREDICTION TYPE BREAKDOWN
# ---------------------------------------------------------
print("PREDICTION TYPE BREAKDOWN:")
print("-" * 40)

type_counts = export_df["prediction_type"].value_counts()

for ptype, count in type_counts.items():
    print(f"  {ptype:20s}: {count:,}")

print(f"\nPower BI Export Completed Successfully!")
