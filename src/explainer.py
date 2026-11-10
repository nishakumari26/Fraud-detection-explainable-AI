# =============================================================
# FILE: src/explainer.py
# PURPOSE: SHAP + LangChain Explainable AI Engine
# =============================================================
#
# THIS IS THE MOST IMPORTANT FILE IN YOUR PROJECT!
# ------------------------------------------------
# This is what separates your project from every other fresher's
# fraud detection project. Here's what it does:
#
# 1. Takes a transaction's features
# 2. Uses SHAP to find WHICH features caused the prediction
# 3. Uses LangChain + Google Gemini to explain it in ENGLISH
#
# WHAT IS SHAP?
# -------------
# SHAP (SHapley Additive exPlanations) is a technique that
# tells you HOW MUCH each feature contributed to a prediction.
#
# Example: If the model says "87% fraud probability", SHAP says:
#   - V14 pushed the probability UP by 0.25 (biggest factor)
#   - V10 pushed it UP by 0.18
#   - Hour pushed it UP by 0.08
#   - Amount pushed it DOWN by 0.03
#
# WHAT IS LangChain?
# ------------------
# LangChain is a framework that makes it easy to work with
# Large Language Models (LLMs) like Google Gemini.
# Instead of just showing numbers, we ask Gemini to write
# a plain-English explanation that a bank analyst can understand.
#
# WHAT IS GOOGLE GEMINI?
# ----------------------
# Gemini is Google's AI model (like ChatGPT but from Google).
# It has a FREE tier — you can make ~60 requests per minute
# without paying anything. Perfect for a portfolio project!
#
# HOW TO RUN:
# python src/explainer.py
# =============================================================

# ---------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------

# numpy is for numerical operations (arrays, math)
import numpy as np

# pandas is for working with tabular data (like spreadsheets)
import pandas as pd

# shap is the explainability library
# It calculates how much each feature contributed to a prediction
import shap

# joblib is for loading our saved model and scaler
import joblib

# os helps us work with file paths and environment variables
import os

# json for converting data structures to/from JSON format
import json

# warnings — we suppress some non-critical warnings
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# TRY TO IMPORT LangChain + Google GenAI
# ---------------------------------------------------------
# We use try/except because:
# 1. The user might not have installed langchain yet
# 2. The user might not have a Gemini API key
# In both cases, we fall back to a template-based explanation
# ---------------------------------------------------------

try:
    # ChatGoogleGenerativeAI is LangChain's wrapper around Gemini
    from langchain_google_genai import ChatGoogleGenerativeAI

    # PromptTemplate helps us create structured prompts
    # In LangChain v1.3+, this moved to langchain_core
    from langchain_core.prompts import PromptTemplate

    # dotenv loads environment variables from a .env file
    from dotenv import load_dotenv

    # Load the .env file (where GOOGLE_API_KEY is stored)
    # We use an explicit path so it works regardless of the
    # current working directory (Streamlit, API, or direct run)
    _env_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        ".env"
    )
    load_dotenv(_env_path)

    # Check if the API key exists
    LANGCHAIN_AVAILABLE = (
        os.getenv("GOOGLE_API_KEY") is not None
    )

    if LANGCHAIN_AVAILABLE:
        print("[OK] LangChain + Gemini API loaded successfully")
    else:
        print("[WARN] GOOGLE_API_KEY not found in .env file")
        print("   Using template-based explanations instead")
        print("   To enable AI explanations:")
        print("   1. Get free key: https://aistudio.google.com/apikey")
        print("   2. Create .env file with: GOOGLE_API_KEY=your_key")

except ImportError:

    LANGCHAIN_AVAILABLE = False

    print("[WARN] LangChain not installed")
    print("   Using template-based explanations instead")
    print("   To install: pip install langchain-google-genai python-dotenv")


# =============================================================
# FRAUD EXPLAINER CLASS
# =============================================================

class FraudExplainer:
    """
    Explains fraud predictions using SHAP + LangChain.

    This class:
    1. Loads the trained XGBoost model
    2. Creates a SHAP explainer for feature attribution
    3. Uses LangChain + Gemini for natural language explanations
    4. Falls back to template-based explanations if no API key

    Usage:
        explainer = FraudExplainer()
        result = explainer.explain_transaction(features_array)
        print(result["explanation"])
    """

    def __init__(self):
        """
        Initialize the explainer by loading the model
        and creating the SHAP explainer.
        """

        # Find the project root directory
        # __file__ = path to this file (explainer.py)
        # We go up twice: src/ → project root
        self.base_dir = os.path.dirname(
            os.path.dirname(
                os.path.abspath(__file__)
            )
        )

        # Load the trained XGBoost model
        self.model = joblib.load(
            os.path.join(
                self.base_dir,
                "artifacts/model_best.pkl"
            )
        )

        # Load the scaler (for preprocessing new transactions)
        self.scaler = joblib.load(
            os.path.join(
                self.base_dir,
                "artifacts/scaler.pkl"
            )
        )

        # Load the optimal threshold
        self.threshold = joblib.load(
            os.path.join(
                self.base_dir,
                "artifacts/optimal_threshold.pkl"
            )
        )

        # Load feature names
        self.feature_names = joblib.load(
            os.path.join(
                self.base_dir,
                "artifacts/feature_names.pkl"
            )
        )

        # -------------------------------------------------
        # CREATE SHAP EXPLAINER
        # -------------------------------------------------
        # TreeExplainer is optimized for tree-based models
        # like XGBoost, Random Forest, LightGBM.
        # It's MUCH faster than the generic KernelExplainer.
        # -------------------------------------------------
        self.shap_explainer = shap.TreeExplainer(
            self.model
        )

        # -------------------------------------------------
        # SET UP LangChain (if available)
        # -------------------------------------------------
        if LANGCHAIN_AVAILABLE:
            self._setup_langchain()

        print("[OK] FraudExplainer initialized successfully")

    def _setup_langchain(self):
        """
        Set up the LangChain pipeline with Google Gemini.

        This creates:
        1. A Gemini LLM instance
        2. A prompt template that tells Gemini how to explain
        """

        # Create the Gemini model instance
        # temperature=0.3 means "be mostly factual, slightly creative"
        # (0.0 = completely factual, 1.0 = very creative)
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-3-flash-preview",
            temperature=0.3
        )

        # Create the prompt template
        # This is the instruction we send to Gemini
        # {variables} get replaced with actual values
        self.prompt_template = PromptTemplate(
            input_variables=[
                "fraud_probability",
                "risk_level",
                "top_features",
                "threshold"
            ],
            template="""You are a financial fraud analyst AI assistant. 
A credit card transaction has been analyzed by a machine learning model.

PREDICTION RESULTS:
- Fraud Probability: {fraud_probability}%
- Risk Level: {risk_level}
- Decision Threshold: {threshold}

TOP CONTRIBUTING FEATURES (from SHAP analysis):
{top_features}

CONTEXT ABOUT THE FEATURES:
- V1 through V28 are PCA-transformed features from the original transaction data (anonymized for privacy)
- Negative values in V14, V12, V10, V17 are strong fraud indicators
- Amount_log is the log-transformed transaction amount
- Hour is the hour of day (0-23) when the transaction occurred
- Features with large absolute SHAP values had the most influence on the prediction

Write a clear, professional explanation (3-5 sentences) for a bank fraud analyst explaining:
1. Whether this transaction is likely fraudulent and why
2. Which factors contributed most to this assessment
3. A recommended action (approve, flag for review, or block)

Keep the language professional but easy to understand. Do NOT use technical jargon like "SHAP values" or "PCA" — explain in business terms."""
        )

        print("[OK] LangChain prompt template configured")

    def get_shap_values(self, X_scaled):
        """
        Calculate SHAP values for a single transaction.

        WHAT ARE SHAP VALUES?
        ---------------------
        For each feature, SHAP gives a number that says:
        - Positive value → this feature INCREASED fraud probability
        - Negative value → this feature DECREASED fraud probability
        - Larger absolute value → more important feature

        Parameters:
        -----------
        X_scaled : numpy array, shape (1, 30)
            The scaled feature values for one transaction

        Returns:
        --------
        dict with:
            - shap_values: array of SHAP values for each feature
            - top_features: dict of top 5 most important features
        """

        # Convert to DataFrame with column names
        # SHAP needs column names to label its output
        X_df = pd.DataFrame(
            X_scaled,
            columns=self.feature_names
        )

        # Calculate SHAP values
        # This tells us how much each feature contributed
        shap_values = self.shap_explainer.shap_values(X_df)

        # If shap_values is a list (for multi-class), take the first
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        # Get the SHAP values for our single transaction
        # shap_values shape: (1, 30) → we want (30,)
        single_shap = shap_values[0]

        # Create a dictionary: feature_name → shap_value
        feature_shap = {}

        for i, name in enumerate(self.feature_names):
            feature_shap[name] = round(
                float(single_shap[i]), 4
            )

        # Sort by absolute value to find the most important
        # key=lambda x: abs(x[1]) sorts by absolute SHAP value
        sorted_features = sorted(
            feature_shap.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )

        # Take top 5 features
        top_features = dict(sorted_features[:5])

        return {
            "shap_values": single_shap,
            "feature_shap": feature_shap,
            "top_features": top_features
        }

    def _get_risk_level(self, fraud_prob):
        """
        Determine risk level based on fraud probability.

        Risk Categories:
        - low: below threshold (legitimate)
        - medium: at or slightly above threshold
        - high: significantly above threshold
        - critical: very high probability

        Parameters:
        -----------
        fraud_prob : float
            Fraud probability (0.0 to 1.0)

        Returns:
        --------
        str : "low", "medium", "high", or "critical"
        """

        if fraud_prob >= 0.8:
            return "critical"

        elif fraud_prob >= 0.5:
            return "high"

        elif fraud_prob >= self.threshold:
            return "medium"

        else:
            return "low"

    def _template_explanation(
        self,
        fraud_prob,
        risk_level,
        top_features
    ):
        """
        Generate a template-based explanation (NO API needed).

        This is the fallback when Gemini is not available.
        It still provides useful explanations using string
        formatting — just not as natural as AI-generated ones.

        Parameters:
        -----------
        fraud_prob : float
            Fraud probability (0.0 to 1.0)

        risk_level : str
            "low", "medium", "high", or "critical"

        top_features : dict
            Top 5 SHAP features and their values

        Returns:
        --------
        str : The explanation text
        """

        # Determine the verdict
        if risk_level in ("critical", "high"):
            verdict = "SUSPICIOUS - likely fraudulent"
            action = "BLOCK this transaction and alert the fraud team"

        elif risk_level == "medium":
            verdict = "UNCERTAIN - requires manual review"
            action = "FLAG for manual review by a fraud analyst"

        else:
            verdict = "LEGITIMATE - appears normal"
            action = "APPROVE this transaction"

        # Build the feature explanation
        feature_lines = []

        for feature, shap_val in top_features.items():

            # Determine direction
            direction = (
                "increased fraud risk"
                if shap_val > 0
                else "decreased fraud risk"
            )

            feature_lines.append(
                f"  - {feature}: SHAP value = {shap_val:+.4f} "
                f"({direction})"
            )

        features_text = "\n".join(feature_lines)

        # Build the full explanation
        explanation = f"""
FRAUD ANALYSIS REPORT
{'=' * 40}

VERDICT: {verdict}
Fraud Probability: {fraud_prob*100:.1f}%
Risk Level: {risk_level.upper()}

KEY CONTRIBUTING FACTORS:
{features_text}

RECOMMENDED ACTION: {action}

NOTE: This analysis is based on historical transaction
patterns. Features V1-V28 are anonymized PCA components
from the original credit card transaction data.
""".strip()

        return explanation

    def _langchain_explanation(
        self,
        fraud_prob,
        risk_level,
        top_features
    ):
        """
        Generate an AI-powered explanation using LangChain + Gemini.

        This is the PREMIUM explanation — natural, conversational,
        and sounds like a real fraud analyst wrote it.

        Parameters:
        -----------
        fraud_prob : float
            Fraud probability (0.0 to 1.0)

        risk_level : str
            "low", "medium", "high", or "critical"

        top_features : dict
            Top 5 SHAP features and their values

        Returns:
        --------
        str : AI-generated explanation
        """

        # Format the top features for the prompt
        features_text = ""

        for feature, shap_val in top_features.items():

            direction = (
                "INCREASED fraud risk"
                if shap_val > 0
                else "DECREASED fraud risk"
            )

            features_text += (
                f"- {feature}: contribution = {shap_val:+.4f} "
                f"({direction})\n"
            )

        # Fill in the prompt template with actual values
        prompt = self.prompt_template.format(
            fraud_probability=f"{fraud_prob*100:.1f}",
            risk_level=risk_level.upper(),
            top_features=features_text,
            threshold=f"{self.threshold:.2f}"
        )

        try:
            # Send the prompt to Gemini and get the response
            # We retry up to 3 times if we hit rate limits
            import time as _time

            max_retries = 3

            for attempt in range(max_retries):

                try:
                    # .invoke() sends the request and waits
                    response = self.llm.invoke(prompt)

                    # Extract the text from the response.
                    # Newer langchain-google-genai versions return
                    # .content as a list of content blocks like:
                    # [{'type': 'text', 'text': '...', 'extras': {...}}]
                    # Older versions return a plain string.
                    content = response.content

                    if isinstance(content, list):
                        # New format: extract 'text' from each block
                        content = "".join(
                            block.get("text", "")
                            if isinstance(block, dict)
                            else str(block)
                            for block in content
                        )

                    return content

                except Exception as retry_err:

                    error_str = str(retry_err)

                    # Check if it's a rate limit error (429)
                    if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:

                        # Wait before retrying
                        # 5s, 10s, 20s (exponential backoff)
                        wait_time = 5 * (2 ** attempt)

                        if attempt < max_retries - 1:
                            print(
                                f"   Rate limited. "
                                f"Retrying in {wait_time}s... "
                                f"(attempt {attempt+1}/{max_retries})"
                            )
                            _time.sleep(wait_time)
                        else:
                            # Last attempt failed, raise to outer
                            raise retry_err
                    else:
                        # Not a rate limit error, don't retry
                        raise retry_err

        except Exception as e:

            # If Gemini fails after all retries, fall back
            print(f"[WARN] Gemini API error: {e}")
            print("   Falling back to template explanation")

            return self._template_explanation(
                fraud_prob,
                risk_level,
                top_features
            )

    def explain_transaction(self, X_raw, pre_scaled=False):
        """
        MAIN METHOD — Explain a single transaction.

        This is the method you'll call most often.
        It does everything: prediction, SHAP, explanation.

        Parameters:
        -----------
        X_raw : numpy array, list, or DataFrame, shape (1, 30)
            Feature values for one transaction.
            Can be raw (unscaled) or pre-scaled data.

        pre_scaled : bool, default False
            If False, the scaler will be applied to X_raw
            before prediction (use this for raw input data).
            If True, X_raw is assumed to already be scaled
            (use this when passing processed test data or
            data already transformed by prepare_features).

        Returns:
        --------
        dict with:
            - fraud_prob: float (0.0 to 1.0)
            - is_fraud: bool
            - risk_level: str
            - top_features: dict (top 5 SHAP features)
            - explanation: str (plain English explanation)
            - all_shap_values: dict (all feature SHAP values)
        """

        # Ensure it's a 2D array (model expects shape (1, 30))
        if isinstance(X_raw, pd.DataFrame):
            X = X_raw.values
        else:
            X = np.array(X_raw)

        if X.ndim == 1:
            X = X.reshape(1, -1)

        # -------------------------------------------------
        # STEP 0: Scale features if needed
        # -------------------------------------------------
        # If the data hasn't been scaled yet, apply the
        # same scaler that was fitted during preprocessing.
        # This ensures consistency with training data.
        if not pre_scaled:
            X = self.scaler.transform(X)

        # -------------------------------------------------
        # STEP 1: Get the model's prediction
        # -------------------------------------------------
        # predict_proba returns probabilities for each class
        # [:,1] gets the probability of class 1 (fraud)
        fraud_prob = float(
            self.model.predict_proba(X)[0][1]
        )

        # Compare with threshold to make the decision
        is_fraud = fraud_prob >= self.threshold

        # Get the risk level
        risk_level = self._get_risk_level(fraud_prob)

        # -------------------------------------------------
        # STEP 2: Calculate SHAP values
        # -------------------------------------------------
        shap_result = self.get_shap_values(X)

        top_features = shap_result["top_features"]

        # -------------------------------------------------
        # STEP 3: Generate explanation
        # -------------------------------------------------
        if LANGCHAIN_AVAILABLE:

            explanation = self._langchain_explanation(
                fraud_prob,
                risk_level,
                top_features
            )

        else:

            explanation = self._template_explanation(
                fraud_prob,
                risk_level,
                top_features
            )

        # -------------------------------------------------
        # STEP 4: Return everything
        # -------------------------------------------------
        return {
            "fraud_prob": round(fraud_prob, 4),
            "is_fraud": is_fraud,
            "risk_level": risk_level,
            "top_features": top_features,
            "explanation": explanation,
            "all_shap_values": shap_result["feature_shap"]
        }


# =============================================================
# MAIN — TEST THE EXPLAINER
# =============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("TESTING FRAUD EXPLAINER")
    print("=" * 60)

    # Create the explainer
    explainer = FraudExplainer()

    # ---------------------------------------------------------
    # Load test data to get real transactions
    # ---------------------------------------------------------
    print("\nLoading test data...")

    base_dir = os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )

    X_test = pd.read_csv(
        os.path.join(base_dir, "data/processed/X_test.csv")
    )

    y_test = pd.read_csv(
        os.path.join(base_dir, "data/processed/y_test.csv")
    ).squeeze()

    # ---------------------------------------------------------
    # TEST 1: Explain a FRAUD transaction
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("TEST 1: EXPLAINING A FRAUD TRANSACTION")
    print("=" * 60)

    # Find actual fraud cases in the test set
    fraud_indices = np.where(y_test.values == 1)[0]

    if len(fraud_indices) > 0:

        # Take the first fraud transaction
        fraud_idx = fraud_indices[0]

        fraud_features = X_test.iloc[[fraud_idx]]

        print(f"\nAnalyzing transaction at index {fraud_idx}...")
        print(f"Actual label: FRAUD")

        # Get the explanation
        # pre_scaled=True because X_test is already scaled
        result = explainer.explain_transaction(
            fraud_features, pre_scaled=True
        )

        print(f"\nFraud Probability : {result['fraud_prob']:.4f}")
        print(f"Is Fraud          : {result['is_fraud']}")
        print(f"Risk Level        : {result['risk_level']}")

        print(f"\nTop Contributing Features:")
        for feat, val in result['top_features'].items():
            print(f"  {feat}: {val:+.4f}")

        print(f"\n{'-' * 50}")
        print("AI EXPLANATION:")
        print('-' * 50)
        print(result['explanation'])

    # ---------------------------------------------------------
    # TEST 2: Explain a LEGITIMATE transaction
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("TEST 2: EXPLAINING A LEGITIMATE TRANSACTION")
    print("=" * 60)

    # Find legitimate cases
    legit_indices = np.where(y_test.values == 0)[0]

    if len(legit_indices) > 0:

        legit_idx = legit_indices[0]

        legit_features = X_test.iloc[[legit_idx]]

        print(f"\nAnalyzing transaction at index {legit_idx}...")
        print(f"Actual label: LEGITIMATE")

        result = explainer.explain_transaction(
            legit_features, pre_scaled=True
        )

        print(f"\nFraud Probability : {result['fraud_prob']:.4f}")
        print(f"Is Fraud          : {result['is_fraud']}")
        print(f"Risk Level        : {result['risk_level']}")

        print(f"\nTop Contributing Features:")
        for feat, val in result['top_features'].items():
            print(f"  {feat}: {val:+.4f}")

        print(f"\n{'-' * 50}")
        print("AI EXPLANATION:")
        print('-' * 50)
        print(result['explanation'])

    # ---------------------------------------------------------
    # SUCCESS
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("EXPLAINER TEST COMPLETED!")
    print("=" * 60)

    if LANGCHAIN_AVAILABLE:
        print("\n[OK] Using AI-powered explanations (Gemini)")
    else:
        print("\n[INFO] Using template-based explanations")
        print("   Set GOOGLE_API_KEY for AI explanations")
