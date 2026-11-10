# =============================================================
# FILE: dashboard/dashboard.py
# PURPOSE: Streamlit interactive fraud detection dashboard
# =============================================================
#
# WHAT IS STREAMLIT?
# ------------------
# Streamlit is a Python library that turns Python scripts into
# beautiful web applications. You write Python → it shows up
# as a website. No HTML/CSS/JavaScript needed!
#
# HOW TO RUN:
# streamlit run dashboard/dashboard.py
#
# This opens a web browser at http://localhost:8501
# =============================================================

# ---------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------

# Streamlit — the web dashboard framework
import streamlit as st

# Data manipulation
import pandas as pd
import numpy as np

# Plotting libraries
import plotly.express as px
import plotly.graph_objects as go

# For loading trained model artifacts
import joblib

# SHAP for explainability
import shap

# Matplotlib for SHAP waterfall plots
import matplotlib.pyplot as plt

# System utilities
import os
import sys
import json
from datetime import datetime

# ---------------------------------------------------------
# ADD PROJECT ROOT TO PATH
# ---------------------------------------------------------
# This lets us import our custom modules (database, explainer)
# even though they're in a different directory

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

# Add the project root to Python's search path
sys.path.insert(0, BASE_DIR)

# Now we can import our custom modules
from src.database import FraudDatabase
from src.explainer import FraudExplainer

# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------
# This MUST be the first Streamlit command in the script
# It sets the page title, icon, and layout

st.set_page_config(
    page_title="Fraud Detection XAI Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# CUSTOM CSS STYLING
# ---------------------------------------------------------
# We inject custom CSS to make the dashboard look professional
# This is like adding makeup to the default Streamlit look

st.markdown("""
<style>
    /* Main background and text */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }

    /* Metric cards styling */
    div[data-testid="metric-container"] {
        background-color: #1e1e2e;
        border: 1px solid #313244;
        border-radius: 12px;
        padding: 15px 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }

    div[data-testid="metric-container"] label {
        color: #cdd6f4 !important;
    }

    div[data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #89b4fa !important;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #1e1e2e;
        border-right: 1px solid #313244;
    }

    /* Headers */
    h1, h2, h3 {
        color: #cdd6f4;
    }

    /* Success/error message boxes */
    .stAlert {
        border-radius: 10px;
    }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 20px;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# LOAD RESOURCES (cached for performance)
# ---------------------------------------------------------
# @st.cache_resource tells Streamlit:
# "Load this ONCE and reuse it. Don't reload on every click."
# This makes the dashboard fast and responsive.

@st.cache_resource
def load_model():
    """Load the trained XGBoost model."""
    return joblib.load(
        os.path.join(BASE_DIR, "artifacts/model_best.pkl")
    )

@st.cache_resource
def load_scaler():
    """Load the data scaler."""
    return joblib.load(
        os.path.join(BASE_DIR, "artifacts/scaler.pkl")
    )

@st.cache_resource
def load_threshold():
    """Load the optimal threshold."""
    return joblib.load(
        os.path.join(BASE_DIR, "artifacts/optimal_threshold.pkl")
    )

@st.cache_resource
def load_feature_names():
    """Load the feature names."""
    return joblib.load(
        os.path.join(BASE_DIR, "artifacts/feature_names.pkl")
    )

@st.cache_resource
def load_explainer():
    """Load our custom FraudExplainer."""
    return FraudExplainer()

@st.cache_resource
def load_database():
    """Load the database connection."""
    return FraudDatabase()

@st.cache_data
def load_test_data():
    """Load test data for sample transactions."""
    X_test = pd.read_csv(
        os.path.join(BASE_DIR, "data/processed/X_test.csv")
    )
    y_test = pd.read_csv(
        os.path.join(BASE_DIR, "data/processed/y_test.csv")
    ).squeeze()
    return X_test, y_test

# Load everything
model = load_model()
scaler = load_scaler()
threshold = load_threshold()
feature_names = load_feature_names()
fraud_explainer = load_explainer()
db = load_database()
X_test, y_test = load_test_data()


# =============================================================
# SIDEBAR — NAVIGATION
# =============================================================

st.sidebar.markdown("# 🛡️ Fraud Detection XAI")
st.sidebar.markdown("---")

# Navigation menu
page = st.sidebar.radio(
    "Navigate to:",
    [
        "🏠 Overview",
        "🔍 Transaction Analyzer",
        "📊 Model Performance",
        "📋 Prediction History"
    ],
    index=0
)

st.sidebar.markdown("---")

st.sidebar.markdown("""
**Built with:**
- XGBoost + SHAP
- LangChain + Gemini
- Streamlit
- SQLite
""")

st.sidebar.markdown("---")

st.sidebar.markdown(
    f"*Threshold: {threshold:.4f}*"
)


# =============================================================
# PAGE 1: OVERVIEW
# =============================================================

if page == "🏠 Overview":

    # Page header
    st.markdown("# 🏠 Fraud Detection Overview")
    st.markdown(
        "Real-time monitoring of the fraud detection system"
    )

    st.markdown("---")

    # -------------------------------------------------
    # KPI METRICS ROW
    # -------------------------------------------------
    # Get statistics from the database
    stats = db.get_summary_stats()

    # Create 4 columns for the KPI cards
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="📊 Total Predictions",
            value=f"{stats['total_predictions']:,}"
        )

    with col2:
        st.metric(
            label="🚨 Fraud Detected",
            value=f"{stats['total_fraud']:,}"
        )

    with col3:
        st.metric(
            label="📈 Fraud Rate",
            value=f"{stats['fraud_rate']}%"
        )

    with col4:
        st.metric(
            label="🎯 Avg Probability",
            value=f"{stats['avg_fraud_prob']:.4f}"
        )

    st.markdown("---")

    # -------------------------------------------------
    # CHARTS ROW
    # -------------------------------------------------
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown("### Risk Level Distribution")

        risk_data = stats["risk_counts"]

        if sum(risk_data.values()) > 0:

            # Create a donut chart using Plotly
            fig = go.Figure(
                data=[go.Pie(
                    labels=list(risk_data.keys()),
                    values=list(risk_data.values()),
                    hole=0.5,
                    marker_colors=[
                        "#a6e3a1",   # low — green
                        "#f9e2af",   # medium — yellow
                        "#fab387",   # high — orange
                        "#f38ba8"    # critical — red
                    ],
                    textinfo="label+percent",
                    textfont_size=14
                )]
            )

            fig.update_layout(
                showlegend=True,
                height=350,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#cdd6f4")
            )

            st.plotly_chart(fig, use_container_width=True)

        else:
            st.info(
                "No predictions yet. Use the Transaction "
                "Analyzer to make some predictions!"
            )

    with chart_col2:
        st.markdown("### Recent Alerts")

        recent = db.get_recent_predictions(limit=10)

        if recent:

            alert_df = pd.DataFrame(recent)[
                ["id", "timestamp", "fraud_prob",
                 "risk_level", "is_fraud"]
            ]

            alert_df.columns = [
                "ID", "Time", "Probability",
                "Risk", "Fraud?"
            ]

            # Format the timestamp to be shorter
            alert_df["Time"] = alert_df["Time"].apply(
                lambda x: x[:19] if x else ""
            )

            # Color code fraud column
            alert_df["Fraud?"] = alert_df["Fraud?"].apply(
                lambda x: "🔴 Yes" if x else "🟢 No"
            )

            st.dataframe(
                alert_df,
                use_container_width=True,
                hide_index=True
            )

        else:
            st.info(
                "No predictions yet. Use the Transaction "
                "Analyzer to get started!"
            )

    # -------------------------------------------------
    # QUICK BATCH ANALYSIS
    # -------------------------------------------------
    st.markdown("---")
    st.markdown("### ⚡ Quick Batch Analysis")
    st.markdown(
        "Run the model on sample transactions from the test set"
    )

    batch_size = st.slider(
        "Number of transactions to analyze:",
        min_value=10,
        max_value=200,
        value=50,
        step=10
    )

    if st.button("🚀 Run Batch Analysis", type="primary"):

        # Pick random transactions from test set
        indices = np.random.choice(
            len(X_test),
            size=batch_size,
            replace=False
        )

        progress = st.progress(0)
        status = st.empty()

        fraud_count = 0

        for i, idx in enumerate(indices):

            # Get features for this transaction
            features = X_test.iloc[[idx]]

            # Get prediction
            prob = float(
                model.predict_proba(features)[0][1]
            )

            is_fraud = prob >= threshold

            # Risk level
            if prob >= 0.8:
                risk = "critical"
            elif prob >= 0.5:
                risk = "high"
            elif prob >= threshold:
                risk = "medium"
            else:
                risk = "low"

            if is_fraud:
                fraud_count += 1

            # Save to database
            db.insert_prediction(
                fraud_prob=prob,
                is_fraud=is_fraud,
                risk_level=risk,
                amount_log=(
                    float(features["Amount_log"].values[0])
                    if "Amount_log" in features.columns
                    else None
                ),
                hour=(
                    float(features["Hour"].values[0])
                    if "Hour" in features.columns
                    else None
                )
            )

            # Update progress bar
            progress.progress((i + 1) / batch_size)
            status.text(
                f"Analyzing transaction {i+1}/{batch_size}..."
            )

        status.text("")
        progress.empty()

        st.success(
            f"✅ Analyzed {batch_size} transactions! "
            f"Found {fraud_count} suspicious. "
            f"Refresh to see updated stats."
        )

        # Force refresh the stats
        st.rerun()


# =============================================================
# PAGE 2: TRANSACTION ANALYZER
# =============================================================

elif page == "🔍 Transaction Analyzer":

    st.markdown("# 🔍 Transaction Analyzer")
    st.markdown(
        "Analyze individual transactions with AI-powered "
        "explainability"
    )

    st.markdown("---")

    # -------------------------------------------------
    # OPTION 1: Use a sample transaction
    # -------------------------------------------------
    st.markdown("### Choose a Transaction to Analyze")

    analysis_mode = st.radio(
        "Select mode:",
        [
            "🎲 Random Fraud Sample",
            "🎲 Random Legitimate Sample",
            "🔢 Enter Transaction Index"
        ],
        horizontal=True
    )

    # Initialize session state for selected index
    # This prevents the random index from changing on
    # every Streamlit rerun (e.g., when clicking Analyze)
    if "selected_idx" not in st.session_state:
        st.session_state.selected_idx = None
        st.session_state.selected_mode = None

    selected_features = None
    actual_label = None

    if analysis_mode == "🎲 Random Fraud Sample":

        fraud_indices = np.where(y_test.values == 1)[0]

        if len(fraud_indices) > 0:

            # Only pick a new random index when the mode
            # changes or user clicks the resample button
            if (st.session_state.selected_mode != "fraud"
                    or st.session_state.selected_idx is None):
                st.session_state.selected_idx = int(
                    np.random.choice(fraud_indices)
                )
                st.session_state.selected_mode = "fraud"

            # Button to pick a different random sample
            if st.button("🔄 Pick Another Fraud Sample"):
                st.session_state.selected_idx = int(
                    np.random.choice(fraud_indices)
                )
                st.rerun()

            random_idx = st.session_state.selected_idx
            selected_features = X_test.iloc[[random_idx]]
            actual_label = "FRAUD"

            st.info(
                f"Selected transaction index: {random_idx} "
                f"(Actual: FRAUD)"
            )

    elif analysis_mode == "🎲 Random Legitimate Sample":

        legit_indices = np.where(y_test.values == 0)[0]

        if len(legit_indices) > 0:

            # Only pick a new random index when the mode
            # changes or user clicks the resample button
            if (st.session_state.selected_mode != "legit"
                    or st.session_state.selected_idx is None):
                st.session_state.selected_idx = int(
                    np.random.choice(legit_indices)
                )
                st.session_state.selected_mode = "legit"

            # Button to pick a different random sample
            if st.button("🔄 Pick Another Legit Sample"):
                st.session_state.selected_idx = int(
                    np.random.choice(legit_indices)
                )
                st.rerun()

            random_idx = st.session_state.selected_idx
            selected_features = X_test.iloc[[random_idx]]
            actual_label = "LEGITIMATE"

            st.info(
                f"Selected transaction index: {random_idx} "
                f"(Actual: LEGITIMATE)"
            )

    else:

        # Reset session state for manual mode
        st.session_state.selected_mode = "manual"

        idx = st.number_input(
            "Enter transaction index:",
            min_value=0,
            max_value=len(X_test) - 1,
            value=0
        )

        selected_features = X_test.iloc[[idx]]
        actual_label = (
            "FRAUD" if y_test.values[idx] == 1
            else "LEGITIMATE"
        )

        st.info(
            f"Selected transaction index: {idx} "
            f"(Actual: {actual_label})"
        )

    # -------------------------------------------------
    # ANALYZE BUTTON
    # -------------------------------------------------
    if st.button("🔍 Analyze Transaction", type="primary"):

        if selected_features is not None:

            with st.spinner("🧠 AI is analyzing this transaction..."):

                # Get the full explanation
                # pre_scaled=True because X_test is already
                # scaled data from data/processed/
                result = fraud_explainer.explain_transaction(
                    selected_features, pre_scaled=True
                )

            # -----------------------------------------
            # RESULTS DISPLAY
            # -----------------------------------------
            st.markdown("---")
            st.markdown("## 📊 Analysis Results")

            # KPI cards for this transaction
            res_col1, res_col2, res_col3, res_col4 = st.columns(4)

            with res_col1:
                st.metric(
                    "Fraud Probability",
                    f"{result['fraud_prob']*100:.1f}%"
                )

            with res_col2:
                if result['is_fraud']:
                    st.metric("Verdict", "🔴 FRAUD")
                else:
                    st.metric("Verdict", "🟢 LEGIT")

            with res_col3:
                risk_emoji = {
                    "low": "🟢",
                    "medium": "🟡",
                    "high": "🟠",
                    "critical": "🔴"
                }
                emoji = risk_emoji.get(result['risk_level'], "")
                st.metric(
                    "Risk Level",
                    f"{emoji} {result['risk_level'].upper()}"
                )

            with res_col4:
                if actual_label:
                    st.metric(
                        "Actual Label",
                        actual_label
                    )

            # -----------------------------------------
            # AI EXPLANATION
            # -----------------------------------------
            st.markdown("---")
            st.markdown("### 🤖 AI Explanation")

            st.markdown(
                f"""<div style="
                    background-color: #1e1e2e;
                    border: 1px solid #313244;
                    border-radius: 12px;
                    padding: 20px;
                    line-height: 1.6;
                    color: #cdd6f4;
                    font-size: 15px;
                ">{result['explanation']}</div>""",
                unsafe_allow_html=True
            )

            # -----------------------------------------
            # TOP SHAP FEATURES
            # -----------------------------------------
            st.markdown("---")

            shap_col1, shap_col2 = st.columns(2)

            with shap_col1:
                st.markdown("### 📊 Top Contributing Features")

                # Create a horizontal bar chart
                feat_names = list(result['top_features'].keys())
                feat_values = list(result['top_features'].values())

                # Color based on direction
                colors = [
                    "#f38ba8" if v > 0 else "#a6e3a1"
                    for v in feat_values
                ]

                fig = go.Figure(
                    go.Bar(
                        x=feat_values,
                        y=feat_names,
                        orientation='h',
                        marker_color=colors,
                        text=[f"{v:+.4f}" for v in feat_values],
                        textposition='outside'
                    )
                )

                fig.update_layout(
                    height=300,
                    xaxis_title="SHAP Value (impact on prediction)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#cdd6f4"),
                    yaxis=dict(autorange="reversed")
                )

                st.plotly_chart(fig, use_container_width=True)

            with shap_col2:
                st.markdown("### 📋 Feature Details")

                # Create a table of all features
                feature_df = pd.DataFrame({
                    "Feature": feat_names,
                    "SHAP Value": feat_values,
                    "Direction": [
                        "⬆️ Increased Risk"
                        if v > 0
                        else "⬇️ Decreased Risk"
                        for v in feat_values
                    ]
                })

                st.dataframe(
                    feature_df,
                    use_container_width=True,
                    hide_index=True
                )

                st.markdown("""
                **How to read SHAP values:**
                - 🔴 **Positive** = pushed toward FRAUD
                - 🟢 **Negative** = pushed toward LEGITIMATE
                - **Larger absolute value** = stronger influence
                """)

            # -----------------------------------------
            # SAVE TO DATABASE
            # -----------------------------------------
            db.insert_prediction(
                fraud_prob=result['fraud_prob'],
                is_fraud=result['is_fraud'],
                risk_level=result['risk_level'],
                top_features=result['top_features'],
                ai_explanation=result['explanation'],
                amount_log=(
                    float(selected_features["Amount_log"].values[0])
                    if "Amount_log" in selected_features.columns
                    else None
                ),
                hour=(
                    float(selected_features["Hour"].values[0])
                    if "Hour" in selected_features.columns
                    else None
                )
            )

            st.success("✅ Prediction saved to database!")


# =============================================================
# PAGE 3: MODEL PERFORMANCE
# =============================================================

elif page == "📊 Model Performance":

    st.markdown("# 📊 Model Performance")
    st.markdown(
        "Visualize model evaluation metrics and explainability"
    )

    st.markdown("---")

    # -------------------------------------------------
    # DISPLAY EXISTING PLOTS
    # -------------------------------------------------
    plots_dir = os.path.join(BASE_DIR, "plots")

    # Tab layout for different plot categories
    tab1, tab2, tab3 = st.tabs([
        "📈 Model Metrics",
        "🔍 SHAP Analysis",
        "📊 Data Analysis"
    ])

    with tab1:

        st.markdown("### Model Comparison")

        perf_col1, perf_col2 = st.columns(2)

        with perf_col1:
            roc_path = os.path.join(
                plots_dir, "06_roc_curves.png"
            )

            if os.path.exists(roc_path):
                st.image(
                    roc_path,
                    caption="ROC Curves — All Models",
                    use_column_width=True
                )

        with perf_col2:
            pr_path = os.path.join(
                plots_dir, "07_pr_curves.png"
            )

            if os.path.exists(pr_path):
                st.image(
                    pr_path,
                    caption="Precision-Recall Curves",
                    use_column_width=True
                )

        st.markdown("---")

        perf_col3, perf_col4 = st.columns(2)

        with perf_col3:
            cm_path = os.path.join(
                plots_dir, "08_confusion_matrix.png"
            )

            if os.path.exists(cm_path):
                st.image(
                    cm_path,
                    caption="Confusion Matrix",
                    use_column_width=True
                )

        with perf_col4:
            fi_path = os.path.join(
                plots_dir, "09_feature_importance.png"
            )

            if os.path.exists(fi_path):
                st.image(
                    fi_path,
                    caption="Feature Importance",
                    use_column_width=True
                )

        st.markdown("---")

        thresh_path = os.path.join(
            plots_dir, "13_threshold_analysis.png"
        )

        if os.path.exists(thresh_path):
            st.image(
                thresh_path,
                caption="Business Threshold Optimization",
                use_column_width=True
            )

    with tab2:

        st.markdown("### SHAP Explainability Analysis")

        shap_col1, shap_col2 = st.columns(2)

        with shap_col1:
            bar_path = os.path.join(
                plots_dir, "10_shap_summary_bar.png"
            )

            if os.path.exists(bar_path):
                st.image(
                    bar_path,
                    caption="SHAP Feature Importance (Global)",
                    use_column_width=True
                )

        with shap_col2:
            bee_path = os.path.join(
                plots_dir, "11_shap_beeswarm.png"
            )

            if os.path.exists(bee_path):
                st.image(
                    bee_path,
                    caption="SHAP Beeswarm Plot",
                    use_column_width=True
                )

        st.markdown("---")
        st.markdown("### Individual Fraud Case Explanations")

        wf_cols = st.columns(3)

        for i in range(3):
            wf_path = os.path.join(
                plots_dir,
                f"12_shap_waterfall_{i+1}.png"
            )

            if os.path.exists(wf_path):
                with wf_cols[i]:
                    st.image(
                        wf_path,
                        caption=f"Fraud Case #{i+1}",
                        use_column_width=True
                    )

    with tab3:

        st.markdown("### Dataset Analysis")

        data_col1, data_col2 = st.columns(2)

        with data_col1:
            class_path = os.path.join(
                plots_dir, "01_class_distribution.png"
            )

            if os.path.exists(class_path):
                st.image(
                    class_path,
                    caption="Class Distribution",
                    use_column_width=True
                )

        with data_col2:
            corr_path = os.path.join(
                plots_dir, "02_feature_correlations.png"
            )

            if os.path.exists(corr_path):
                st.image(
                    corr_path,
                    caption="Feature Correlations",
                    use_column_width=True
                )

        st.markdown("---")

        data_col3, data_col4 = st.columns(2)

        with data_col3:
            amt_path = os.path.join(
                plots_dir, "03_amount_distribution.png"
            )

            if os.path.exists(amt_path):
                st.image(
                    amt_path,
                    caption="Amount Distribution",
                    use_column_width=True
                )

        with data_col4:
            hour_path = os.path.join(
                plots_dir, "04_hour_analysis.png"
            )

            if os.path.exists(hour_path):
                st.image(
                    hour_path,
                    caption="Fraud by Hour",
                    use_column_width=True
                )

        feat_path = os.path.join(
            plots_dir, "05_feature_distributions.png"
        )

        if os.path.exists(feat_path):
            st.image(
                feat_path,
                caption="Key Feature Distributions",
                use_column_width=True
            )


# =============================================================
# PAGE 4: PREDICTION HISTORY
# =============================================================

elif page == "📋 Prediction History":

    st.markdown("# 📋 Prediction History")
    st.markdown(
        "View and filter all predictions stored in the database"
    )

    st.markdown("---")

    # -------------------------------------------------
    # FILTERS
    # -------------------------------------------------
    filter_col1, filter_col2, filter_col3 = st.columns(3)

    with filter_col1:
        risk_filter = st.selectbox(
            "Filter by Risk Level:",
            ["All", "critical", "high", "medium", "low"]
        )

    with filter_col2:
        limit = st.number_input(
            "Number of records:",
            min_value=10,
            max_value=1000,
            value=100,
            step=10
        )

    with filter_col3:
        st.markdown("&nbsp;")  # spacer

        if st.button("🗑️ Clear All History"):
            db.clear_all()
            st.success("History cleared!")
            st.rerun()

    # -------------------------------------------------
    # FETCH AND DISPLAY DATA
    # -------------------------------------------------
    if risk_filter == "All":
        predictions = db.get_recent_predictions(limit=limit)
    else:
        predictions = db.get_predictions_by_risk(risk_filter)

    if predictions:

        # Convert to DataFrame
        df = pd.DataFrame(predictions)

        # -----------------------------------------
        # SUMMARY METRICS
        # -----------------------------------------
        sum_col1, sum_col2, sum_col3 = st.columns(3)

        with sum_col1:
            st.metric(
                "Records Found",
                len(df)
            )

        with sum_col2:
            fraud_pct = (
                df["is_fraud"].mean() * 100
                if len(df) > 0 else 0
            )
            st.metric(
                "Fraud Rate",
                f"{fraud_pct:.1f}%"
            )

        with sum_col3:
            avg_prob = (
                df["fraud_prob"].mean()
                if len(df) > 0 else 0
            )
            st.metric(
                "Avg Probability",
                f"{avg_prob:.4f}"
            )

        st.markdown("---")

        # -----------------------------------------
        # DATA TABLE
        # -----------------------------------------
        st.markdown("### 📋 Prediction Records")

        # Select columns to display
        display_cols = [
            "id", "timestamp", "fraud_prob",
            "is_fraud", "risk_level"
        ]

        if "amount_log" in df.columns:
            display_cols.append("amount_log")

        if "hour" in df.columns:
            display_cols.append("hour")

        display_df = df[
            [c for c in display_cols if c in df.columns]
        ].copy()

        # Format columns
        if "timestamp" in display_df.columns:
            display_df["timestamp"] = display_df["timestamp"].apply(
                lambda x: x[:19] if isinstance(x, str) else x
            )

        if "is_fraud" in display_df.columns:
            display_df["is_fraud"] = display_df["is_fraud"].apply(
                lambda x: "🔴 Yes" if x else "🟢 No"
            )

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )

        # -----------------------------------------
        # DOWNLOAD BUTTON
        # -----------------------------------------
        st.markdown("---")

        csv_data = df.to_csv(index=False)

        st.download_button(
            label="📥 Download as CSV",
            data=csv_data,
            file_name="fraud_predictions_export.csv",
            mime="text/csv"
        )

        # -----------------------------------------
        # PROBABILITY DISTRIBUTION CHART
        # -----------------------------------------
        st.markdown("---")
        st.markdown("### 📊 Probability Distribution")

        fig = px.histogram(
            df,
            x="fraud_prob",
            nbins=30,
            color="risk_level",
            color_discrete_map={
                "low": "#a6e3a1",
                "medium": "#f9e2af",
                "high": "#fab387",
                "critical": "#f38ba8"
            },
            labels={
                "fraud_prob": "Fraud Probability",
                "risk_level": "Risk Level"
            }
        )

        fig.update_layout(
            height=350,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#cdd6f4"),
            xaxis=dict(gridcolor="#313244"),
            yaxis=dict(gridcolor="#313244")
        )

        st.plotly_chart(fig, use_container_width=True)

        # -----------------------------------------
        # VIEW EXPLANATION
        # -----------------------------------------
        st.markdown("---")
        st.markdown("### 🤖 View AI Explanation")

        explanation_records = [
            p for p in predictions
            if p.get("ai_explanation")
        ]

        if explanation_records:

            selected_id = st.selectbox(
                "Select prediction ID:",
                [p["id"] for p in explanation_records]
            )

            selected_pred = next(
                p for p in explanation_records
                if p["id"] == selected_id
            )

            st.markdown(
                f"""<div style="
                    background-color: #1e1e2e;
                    border: 1px solid #313244;
                    border-radius: 12px;
                    padding: 20px;
                    line-height: 1.6;
                    color: #cdd6f4;
                ">{selected_pred['ai_explanation']}</div>""",
                unsafe_allow_html=True
            )

        else:
            st.info(
                "No AI explanations saved yet. "
                "Use the Transaction Analyzer to generate some!"
            )

    else:
        st.info(
            "No predictions found. Use the Overview page's "
            "Batch Analysis or the Transaction Analyzer to "
            "create predictions!"
        )
