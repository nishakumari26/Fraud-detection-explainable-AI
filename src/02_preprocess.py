# =============================================================
# FILE: src/02_preprocess.py
# PURPOSE: Professional preprocessing pipeline
# RUN: python src/02_preprocess.py
# =============================================================

import pandas as pd
import numpy as np
import os
import joblib
import warnings

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

warnings.filterwarnings("ignore")

# -------------------------------------------------------------
# CREATE DIRECTORIES
# -------------------------------------------------------------
os.makedirs("data/processed", exist_ok=True)
os.makedirs("artifacts", exist_ok=True)

# -------------------------------------------------------------
# 1. LOAD DATA
# -------------------------------------------------------------
print("=" * 60)
print("STEP 1: LOADING DATA")
print("=" * 60)

df = pd.read_csv("data/raw/creditcard.csv")

print(f"\nDataset Shape: {df.shape}")

# -------------------------------------------------------------
# 2. FEATURE ENGINEERING
# -------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 2: FEATURE ENGINEERING")
print("=" * 60)

# Log transform amount
df['Amount_log'] = np.log1p(df['Amount'])

print("\nCreated feature: Amount_log")

# Hour extraction
df['Hour'] = (df['Time'] / 3600) % 24

print("Created feature: Hour")

# Drop old columns
df.drop(['Amount', 'Time'], axis=1, inplace=True)

print("\nDropped original Amount and Time columns")

# -------------------------------------------------------------
# 3. FEATURES & TARGET
# -------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 3: FEATURES & TARGET")
print("=" * 60)

X = df.drop('Class', axis=1)
y = df['Class']

print(f"\nFeature Shape: {X.shape}")
print(f"Target Shape : {y.shape}")

# -------------------------------------------------------------
# 4. TRAIN / VALIDATION / TEST SPLIT
# -------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 4: TRAIN / VALIDATION / TEST SPLIT")
print("=" * 60)

# First split train+temp
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    stratify=y,
    random_state=42
)

# Split temp into validation + test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    stratify=y_temp,
    random_state=42
)

print(f"\nTrain Shape      : {X_train.shape}")
print(f"Validation Shape : {X_val.shape}")
print(f"Test Shape       : {X_test.shape}")

print("\nFraud Rates:")

print(f"Train      : {y_train.mean()*100:.4f}%")
print(f"Validation : {y_val.mean()*100:.4f}%")
print(f"Test       : {y_test.mean()*100:.4f}%")

# -------------------------------------------------------------
# 5. FEATURE SCALING
# -------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 5: FEATURE SCALING")
print("=" * 60)

scaler = StandardScaler()

# FIT ONLY ON TRAIN
X_train_scaled = scaler.fit_transform(X_train)

# ONLY TRANSFORM val/test
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print("\nScaler fitted ONLY on training data")
print("Validation/Test transformed using same scaler")

# Convert back to DataFrame
X_train_scaled = pd.DataFrame(
    X_train_scaled,
    columns=X_train.columns
)

X_val_scaled = pd.DataFrame(
    X_val_scaled,
    columns=X_val.columns
)

X_test_scaled = pd.DataFrame(
    X_test_scaled,
    columns=X_test.columns
)

# -------------------------------------------------------------
# 6. HANDLE CLASS IMBALANCE
# -------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 6: CLASS IMBALANCE HANDLING")
print("=" * 60)

print("\nOriginal Training Distribution:")

print(y_train.value_counts())

# ---------------- SMOTE ----------------
print("\nApplying SMOTE...")

smote = SMOTE(
    random_state=42,
    k_neighbors=5
)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train_scaled,
    y_train
)

print("\nAfter SMOTE:")

print(y_train_smote.value_counts())

# ---------------- UNDERSAMPLING ----------------
print("\nApplying Random Undersampling...")

rus = RandomUnderSampler(random_state=42)

X_train_under, y_train_under = rus.fit_resample(
    X_train_scaled,
    y_train
)

print("\nAfter Undersampling:")

print(y_train_under.value_counts())

# ---------------- COST-SENSITIVE ----------------
scale_pos_weight = (
    (y_train == 0).sum() /
    (y_train == 1).sum()
)

print(f"\nscale_pos_weight = {scale_pos_weight:.2f}")

# -------------------------------------------------------------
# 7. SAVE PROCESSED DATA
# -------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 7: SAVING FILES")
print("=" * 60)

# -------- TRAIN --------
X_train_scaled.to_csv(
    "data/processed/X_train.csv",
    index=False
)

y_train.to_csv(
    "data/processed/y_train.csv",
    index=False
)

# -------- VALIDATION --------
X_val_scaled.to_csv(
    "data/processed/X_val.csv",
    index=False
)

y_val.to_csv(
    "data/processed/y_val.csv",
    index=False
)

# -------- TEST --------
X_test_scaled.to_csv(
    "data/processed/X_test.csv",
    index=False
)

y_test.to_csv(
    "data/processed/y_test.csv",
    index=False
)

# -------- SMOTE --------
pd.DataFrame(
    X_train_smote,
    columns=X_train.columns
).to_csv(
    "data/processed/X_train_smote.csv",
    index=False
)

pd.Series(
    y_train_smote,
    name='Class'
).to_csv(
    "data/processed/y_train_smote.csv",
    index=False
)

# -------- UNDERSAMPLING --------
pd.DataFrame(
    X_train_under,
    columns=X_train.columns
).to_csv(
    "data/processed/X_train_under.csv",
    index=False
)

pd.Series(
    y_train_under,
    name='Class'
).to_csv(
    "data/processed/y_train_under.csv",
    index=False
)

# -------------------------------------------------------------
# 8. SAVE ARTIFACTS
# -------------------------------------------------------------
print("\nSaving artifacts...")

joblib.dump(
    scaler,
    "artifacts/scaler.pkl"
)

joblib.dump(
    scale_pos_weight,
    "artifacts/scale_pos_weight.pkl"
)

joblib.dump(
    list(X_train.columns),
    "artifacts/feature_names.pkl"
)

print("\nArtifacts Saved:")
print("- scaler.pkl")
print("- scale_pos_weight.pkl")
print("- feature_names.pkl")

# -------------------------------------------------------------
# 9. FINAL SUMMARY
# -------------------------------------------------------------
print("\n" + "=" * 60)
print("PREPROCESSING COMPLETED")
print("=" * 60)

print("""
Pipeline Summary:

1. Feature Engineering
   - Amount_log
   - Hour

2. Data Split
   - Train: 70%
   - Validation: 15%
   - Test: 15%

3. Scaling
   - StandardScaler
   - No data leakage

4. Imbalance Handling
   - SMOTE
   - Undersampling
   - Cost-sensitive weighting

5. Saved deployment artifacts
""")

print("\nNext Step: Model Training")