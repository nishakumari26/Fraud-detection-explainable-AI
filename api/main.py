# =============================================================
# FILE: api/main.py
# PURPOSE: Production fraud detection API
# RUN: uvicorn api.main:app --reload
# =============================================================

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import numpy as np
import pandas as pd
import joblib
import os
import time

# -------------------------------------------------------------
# 1. LOAD ARTIFACTS
# -------------------------------------------------------------
BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

print("=" * 60)
print("LOADING MODEL ARTIFACTS")
print("=" * 60)

model = joblib.load(
    os.path.join(
        BASE_DIR,
        "artifacts/model_best.pkl"
    )
)

scaler = joblib.load(
    os.path.join(
        BASE_DIR,
        "artifacts/scaler.pkl"
    )
)

threshold = joblib.load(
    os.path.join(
        BASE_DIR,
        "artifacts/optimal_threshold.pkl"
    )
)

feature_names = joblib.load(
    os.path.join(
        BASE_DIR,
        "artifacts/feature_names.pkl"
    )
)

print(f"\nModel loaded successfully")
print(f"Optimal threshold: {threshold}")

# -------------------------------------------------------------
# 2. CREATE FASTAPI APP
# -------------------------------------------------------------
app = FastAPI(
    title="Credit Card Fraud Detection API",

    description="""
Production-grade fraud detection system using:
- XGBoost
- SMOTE
- Optuna
- SHAP Explainability

Features:
- Real-time fraud prediction
- Batch inference
- Business-threshold optimization
- Production-ready deployment
""",

    version="1.0.0"
)

# -------------------------------------------------------------
# 3. REQUEST MODELS
# -------------------------------------------------------------
class Transaction(BaseModel):

    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float

    Amount_log: float = Field(
        ...,
        description="log(amount + 1)"
    )

    Hour: float = Field(
        ...,
        description="hour of day"
    )

class BatchRequest(BaseModel):
    transactions: List[Transaction]

# -------------------------------------------------------------
# 4. HELPER FUNCTION
# -------------------------------------------------------------
def prepare_features(transaction):

    values = [
        transaction.V1,
        transaction.V2,
        transaction.V3,
        transaction.V4,
        transaction.V5,
        transaction.V6,
        transaction.V7,
        transaction.V8,
        transaction.V9,
        transaction.V10,
        transaction.V11,
        transaction.V12,
        transaction.V13,
        transaction.V14,
        transaction.V15,
        transaction.V16,
        transaction.V17,
        transaction.V18,
        transaction.V19,
        transaction.V20,
        transaction.V21,
        transaction.V22,
        transaction.V23,
        transaction.V24,
        transaction.V25,
        transaction.V26,
        transaction.V27,
        transaction.V28,
        transaction.Amount_log,
        transaction.Hour
    ]

    X = np.array(values).reshape(1, -1)

    X_scaled = scaler.transform(X)

    return X_scaled

# -------------------------------------------------------------
# 5. HEALTH CHECK ENDPOINT
# -------------------------------------------------------------
@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model": "XGBoost Fraud Detection",
        "version": "1.0.0"
    }

# -------------------------------------------------------------
# 6. MODEL INFO ENDPOINT
# -------------------------------------------------------------
@app.get("/model-info")
def model_info():

    return {

        "model_type": "XGBoost Classifier",

        "fraud_rate": "0.17%",

        "metrics": {
            "pr_auc": "0.84",
            "f1_score": "0.77"
        },

        "threshold": float(threshold),

        "features": feature_names,

        "known_limitations": [
            "Trained on 2013 European transactions",
            "No drift monitoring implemented",
            "Requires periodic retraining"
        ]
    }

# -------------------------------------------------------------
# 7. SINGLE PREDICTION ENDPOINT
# -------------------------------------------------------------
@app.post("/predict")
def predict(transaction: Transaction):

    try:

        start = time.time()

        X = prepare_features(transaction)

        prob = float(
            model.predict_proba(X)[0][1]
        )

        prediction = bool(
            prob >= threshold
        )

        # Risk categorization
        if prob >= 0.8:
            risk = "critical"

        elif prob >= 0.5:
            risk = "high"

        elif prob >= threshold:
            risk = "medium"

        else:
            risk = "low"

        latency = (
            time.time() - start
        ) * 1000

        return {

            "fraud_probability": round(prob, 4),

            "is_fraud": prediction,

            "threshold_used": round(
                float(threshold),
                4
            ),

            "risk_level": risk,

            "latency_ms": round(
                latency,
                2
            ),

            "message": (
                "Transaction flagged as fraud"
                if prediction
                else
                "Transaction appears legitimate"
            )
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

# -------------------------------------------------------------
# 8. BATCH PREDICTION ENDPOINT
# -------------------------------------------------------------
@app.post("/predict/batch")
def predict_batch(batch: BatchRequest):

    try:

        start = time.time()

        results = []

        for transaction in batch.transactions:

            X = prepare_features(transaction)

            prob = float(
                model.predict_proba(X)[0][1]
            )

            prediction = bool(
                prob >= threshold
            )

            results.append({

                "fraud_probability": round(prob, 4),

                "is_fraud": prediction
            })

        latency = (
            time.time() - start
        ) * 1000

        fraud_count = sum(
            r["is_fraud"]
            for r in results
        )

        return {

            "total_transactions": len(results),

            "fraud_detected": fraud_count,

            "processing_time_ms": round(
                latency,
                2
            ),

            "results": results
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

# -------------------------------------------------------------
# 9. EXPLAINABLE AI ENDPOINT (NEW!)
# -------------------------------------------------------------
# This is the KEY differentiator of your project.
# It not only predicts fraud but EXPLAINS why in plain English.
# Uses SHAP for feature attribution + LangChain + Gemini
# for natural language explanation.
# -------------------------------------------------------------

# We import here to avoid circular imports and so the API
# still works even if langchain is not installed
try:
    import sys
    sys.path.insert(0, BASE_DIR)
    from src.explainer import FraudExplainer
    from src.database import FraudDatabase

    explainer_engine = FraudExplainer()
    db = FraudDatabase()

    XAI_AVAILABLE = True
    print("\n✅ Explainable AI engine loaded")

except Exception as e:
    XAI_AVAILABLE = False
    print(f"\n⚠️  XAI engine not available: {e}")
    print("   /predict/explain endpoint will be disabled")


@app.post("/predict/explain")
def predict_with_explanation(transaction: Transaction):
    """
    Predict fraud AND explain WHY using SHAP + LangChain.

    This is the premium endpoint that:
    1. Predicts fraud probability
    2. Calculates SHAP feature attributions
    3. Generates a plain-English explanation using Gemini AI
    4. Saves everything to the SQLite database

    Returns the prediction, risk level, top features,
    and the AI-generated explanation.
    """

    if not XAI_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail=(
                "Explainable AI engine not available. "
                "Please check that src/explainer.py and "
                "src/database.py are properly set up."
            )
        )

    try:

        start = time.time()

        # Prepare the features
        X = prepare_features(transaction)

        # Get full explanation (prediction + SHAP + LangChain)
        # pre_scaled=True because prepare_features() already
        # applied scaler.transform() above
        result = explainer_engine.explain_transaction(
            X, pre_scaled=True
        )

        # Save to database for audit trail
        prediction_id = db.insert_prediction(
            fraud_prob=result["fraud_prob"],
            is_fraud=result["is_fraud"],
            risk_level=result["risk_level"],
            top_features=result["top_features"],
            ai_explanation=result["explanation"],
            amount_log=transaction.Amount_log,
            hour=transaction.Hour
        )

        latency = (time.time() - start) * 1000

        return {

            "prediction_id": prediction_id,

            "fraud_probability": round(
                result["fraud_prob"], 4
            ),

            "is_fraud": result["is_fraud"],

            "risk_level": result["risk_level"],

            "threshold_used": round(
                float(threshold), 4
            ),

            "top_contributing_features": result["top_features"],

            "ai_explanation": result["explanation"],

            "latency_ms": round(latency, 2),

            "saved_to_database": True
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# -------------------------------------------------------------
# 10. PREDICTION HISTORY ENDPOINT
# -------------------------------------------------------------

@app.get("/history")
def get_prediction_history(limit: int = 50):
    """
    Retrieve recent predictions from the database.

    Parameters:
    - limit: Number of records to return (default: 50)
    """

    if not XAI_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Database not available"
        )

    predictions = db.get_recent_predictions(limit=limit)

    stats = db.get_summary_stats()

    return {
        "total_in_database": stats["total_predictions"],
        "showing": len(predictions),
        "summary": stats,
        "predictions": predictions
    }


# -------------------------------------------------------------
# 11. MAIN
# -------------------------------------------------------------
if __name__ == "__main__":

    import uvicorn

    print("\n" + "=" * 60)
    print("STARTING FRAUD DETECTION API")
    print("=" * 60)

    print("\nSwagger Docs:")
    print("http://127.0.0.1:8000/docs")

    print("\nEndpoints:")
    print("  POST /predict         — Basic prediction")
    print("  POST /predict/explain — Prediction + AI explanation")
    print("  POST /predict/batch   — Batch predictions")
    print("  GET  /history         — Prediction history")
    print("  GET  /health          — Health check")
    print("  GET  /model-info      — Model information")

    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )