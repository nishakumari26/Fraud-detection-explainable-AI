# =============================================================
# FILE: src/01_eda.py
# PURPOSE: Exploratory Data Analysis
# RUN: python src/01_eda.py
# =============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create plots directory
os.makedirs("plots", exist_ok=True)

# ---------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------
print("=" * 60)
print("STEP 1: LOADING DATA")
print("=" * 60)

df = pd.read_csv("data/raw/creditcard.csv")

print(f"\nDataset Shape: {df.shape}")
print(f"Rows    : {df.shape[0]:,}")
print(f"Columns : {df.shape[1]}")

print("\nFirst 5 rows:")
print(df.head())

# ---------------------------------------------------------
# 2. BASIC DATASET INFORMATION
# ---------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 2: BASIC DATASET INFORMATION")
print("=" * 60)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nStatistical Summary:")
print(df.describe())

# ---------------------------------------------------------
# 3. CLASS IMBALANCE ANALYSIS
# ---------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 3: CLASS IMBALANCE ANALYSIS")
print("=" * 60)

class_counts = df['Class'].value_counts()

legit_count = class_counts[0]
fraud_count = class_counts[1]

fraud_rate = df['Class'].mean() * 100

print(f"\nLegitimate Transactions : {legit_count:,}")
print(f"Fraudulent Transactions : {fraud_count:,}")
print(f"Fraud Rate              : {fraud_rate:.4f}%")

print("\nIMPORTANT INSIGHT:")
print(f"A model predicting ONLY legitimate transactions")
print(f"would still achieve {100-fraud_rate:.2f}% accuracy.")
print("This is why ACCURACY is a misleading metric here.")

# ---------------------------------------------------------
# 4. CLASS DISTRIBUTION PLOT
# ---------------------------------------------------------
plt.figure(figsize=(8,5))

sns.countplot(x='Class', data=df)

plt.title("Transaction Class Distribution")
plt.xticks([0,1], ['Legitimate', 'Fraud'])
plt.xlabel("Transaction Type")
plt.ylabel("Count")

plt.tight_layout()

plt.savefig("plots/01_class_distribution.png", dpi=300)

plt.show()

print("\nSaved: plots/01_class_distribution.png")

# ---------------------------------------------------------
# 5. FRAUD VS LEGIT ANALYSIS
# ---------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 4: FRAUD VS LEGIT COMPARISON")
print("=" * 60)

fraud = df[df['Class'] == 1]
legit = df[df['Class'] == 0]

print("\nMean Comparison for Important Features:\n")

features_to_compare = [
    'V4', 'V10', 'V11', 'V12',
    'V14', 'V17', 'Amount'
]

for col in features_to_compare:

    fraud_mean = fraud[col].mean()
    legit_mean = legit[col].mean()

    print(f"{col}")
    print(f"Fraud Mean : {fraud_mean:.4f}")
    print(f"Legit Mean : {legit_mean:.4f}")
    print("-" * 40)

# ---------------------------------------------------------
# 6. CORRELATION ANALYSIS
# ---------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 5: CORRELATION ANALYSIS")
print("=" * 60)

correlations = df.corr()['Class'].sort_values()

print("\nTop Negative Correlations:")
print(correlations.head(10))

print("\nTop Positive Correlations:")
print(correlations.tail(10))

# ---------------------------------------------------------
# 7. CORRELATION BAR CHART
# ---------------------------------------------------------
top_corr = pd.concat([
    correlations.head(10),
    correlations.tail(10)
])

plt.figure(figsize=(12,7))

colors = ['red' if x > 0 else 'blue'
          for x in top_corr.values]

plt.barh(top_corr.index, top_corr.values, color=colors)

plt.axvline(0, color='black')

plt.title("Feature Correlation with Fraud")
plt.xlabel("Correlation with Class")

plt.tight_layout()

plt.savefig("plots/02_feature_correlations.png", dpi=300)

plt.show()

print("\nSaved: plots/02_feature_correlations.png")

# ---------------------------------------------------------
# 8. TRANSACTION AMOUNT ANALYSIS
# ---------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 6: TRANSACTION AMOUNT ANALYSIS")
print("=" * 60)

print("\nLegitimate Transaction Statistics:")
print(legit['Amount'].describe())

print("\nFraudulent Transaction Statistics:")
print(fraud['Amount'].describe())

# Log transform amount
df['Amount_log'] = np.log1p(df['Amount'])

plt.figure(figsize=(10,6))

sns.histplot(
    data=df,
    x='Amount_log',
    hue='Class',
    bins=50,
    kde=True
)

plt.title("Log-Transformed Transaction Amount Distribution")

plt.tight_layout()

plt.savefig("plots/03_amount_distribution.png", dpi=300)

plt.show()

print("\nSaved: plots/03_amount_distribution.png")

# ---------------------------------------------------------
# 9. TIME FEATURE ANALYSIS
# ---------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 7: TIME ANALYSIS")
print("=" * 60)

# Convert seconds into hour of day
df['Hour'] = (df['Time'] / 3600) % 24

plt.figure(figsize=(12,6))

sns.histplot(
    data=df,
    x='Hour',
    hue='Class',
    bins=24,
    kde=False
)

plt.title("Fraud vs Legit Transactions by Hour")

plt.xlabel("Hour of Day")

plt.tight_layout()

plt.savefig("plots/04_hour_analysis.png", dpi=300)

plt.show()

print("\nSaved: plots/04_hour_analysis.png")

# ---------------------------------------------------------
# 10. FEATURE DISTRIBUTION ANALYSIS
# ---------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 8: FEATURE DISTRIBUTIONS")
print("=" * 60)

important_features = ['V10', 'V12', 'V14', 'V17']

fig, axes = plt.subplots(2, 2, figsize=(14,10))

axes = axes.flatten()

for i, feature in enumerate(important_features):

    axes[i].hist(
        legit[feature],
        bins=50,
        alpha=0.6,
        label='Legit',
        density=True
    )

    axes[i].hist(
        fraud[feature],
        bins=50,
        alpha=0.6,
        label='Fraud',
        density=True
    )

    axes[i].set_title(f"{feature} Distribution")
    axes[i].legend()

plt.tight_layout()

plt.savefig("plots/05_feature_distributions.png", dpi=300)

plt.show()

print("\nSaved: plots/05_feature_distributions.png")

# ---------------------------------------------------------
# 11. FINAL EDA INSIGHTS
# ---------------------------------------------------------
print("\n" + "=" * 60)
print("FINAL EDA INSIGHTS")
print("=" * 60)

print("""
1. Dataset is EXTREMELY imbalanced
   Fraud rate is only 0.17%

2. Accuracy is NOT an appropriate metric

3. Features V10, V12, V14, V17
   show strong fraud separation

4. Fraud transactions tend to have
   different amount distributions

5. Fraud patterns vary by hour of day

6. PR-AUC and Recall are critical metrics
   for this business problem
""")

print("\nEDA COMPLETED SUCCESSFULLY!")
print("\nNext Step: Preprocessing & Feature Engineering")