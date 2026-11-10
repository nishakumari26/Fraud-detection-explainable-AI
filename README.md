# 🛡️ Financial Fraud Detection with Explainable AI

> An end-to-end fraud detection system that not only detects fraud using machine learning but also **explains WHY** a transaction is suspicious using Generative AI, with a live dashboard for analysts.

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-ML-green?logo=xgboost)
![LangChain](https://img.shields.io/badge/LangChain-GenAI-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?logo=streamlit)
![FastAPI](https://img.shields.io/badge/FastAPI-API-teal?logo=fastapi)

---

## 🌟 What Makes This Project Special

| Feature | Most Fresher Projects | This Project |
|---------|----------------------|-------------|
| **Model** | Single model, no tuning | 5 models + Optuna hyperparameter tuning |
| **Explainability** | None or basic SHAP | SHAP + **LangChain + Gemini AI** for plain-English explanations |
| **Data Storage** | None | SQLite database with full audit trail |
| **Dashboard** | None | 4-page Streamlit interactive dashboard |
| **API** | Basic Flask | Production FastAPI with 6 endpoints |
| **Business Value** | "95% accuracy" | Cost-optimized threshold + Power BI export |
| **Deployment** | None | Docker-ready |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│              FRAUD DETECTION WITH EXPLAINABLE AI            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Transaction ──► XGBoost Model ──► SHAP ──► LangChain/Gemini│
│                  "Is it fraud?"    "Why?"   "In English"    │
│                       │               │           │         │
│                       ▼               ▼           ▼         │
│                  SQLite Database (audit trail)              │
│                       │                                     │
│               ┌───────┴───────┐                             │
│               ▼               ▼                             │
│         Streamlit         Power BI                          │
│         Dashboard         Export                            │
└─────────────────────────────────────────────────────────────┘
```

---

## 📊 Results

| Metric | Value |
|--------|-------|
| **Best Model** | XGBoost + Optuna |
| **PR-AUC** | ~0.84 |
| **ROC-AUC** | ~0.98 |
| **Fraud Detection Rate** | Optimized via business cost model |
| **Dataset** | 284,807 transactions (0.17% fraud) |

---

## 🛠️ Tech Stack

| Category | Technology |
|----------|-----------|
| **ML/DL** | XGBoost, Scikit-learn, Optuna, SMOTE |
| **Explainability** | SHAP, LangChain, Google Gemini |
| **API** | FastAPI, Uvicorn, Pydantic |
| **Dashboard** | Streamlit, Plotly |
| **Database** | SQLite |
| **Visualization** | Matplotlib, Seaborn, Plotly |
| **Data** | Pandas, NumPy |
| **Deployment** | Docker |

---

## 📁 Project Structure

```
fraud-detection/
│
├── data/
│   ├── raw/creditcard.csv              # Original Kaggle dataset
│   └── processed/                      # Train/val/test splits
│
├── src/
│   ├── 01_eda.py                       # Exploratory Data Analysis
│   ├── 02_preprocess.py                # Feature engineering + SMOTE
│   ├── 03_train.py                     # Model training + Optuna
│   ├── 04_evaluate.py                  # Evaluation + SHAP
│   ├── 05_threshold.py                 # Business cost optimization
│   ├── database.py                     # SQLite database layer
│   ├── explainer.py                    # SHAP + LangChain XAI engine
│   └── export_powerbi.py              # Power BI data export
│
├── dashboard/
│   └── dashboard.py                    # Streamlit web dashboard
│
├── api/
│   └── main.py                         # FastAPI REST API
│
├── artifacts/                          # Trained models & scaler
├── plots/                              # 15+ analysis plots
├── powerbi/                            # Power BI export CSVs
│
├── requirements.txt
├── Dockerfile
├── .env.example
├── .gitignore
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- pip

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/fraud-detection.git
cd fraud-detection
```

### 2. Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. (Optional) Set Up Gemini API Key

Get a **FREE** API key from [Google AI Studio](https://aistudio.google.com/apikey), then:

```bash
# Create .env file
copy .env.example .env

# Edit .env and add your key
# GOOGLE_API_KEY=your_key_here
```

> **Note:** The project works perfectly WITHOUT an API key — it just uses template-based explanations instead of AI-generated ones.

### 5. Run the Pipeline (if starting fresh)

```bash
# Step 1: Download creditcard.csv from Kaggle and place in data/raw/
# Step 2: Run each script in order
python src/01_eda.py
python src/02_preprocess.py
python src/03_train.py
python src/04_evaluate.py
python src/05_threshold.py
```

### 6. Launch the Dashboard

```bash
streamlit run dashboard/dashboard.py
```

### 7. Launch the API

```bash
python api/main.py
# Visit: http://127.0.0.1:8000/docs
```

### 8. Export for Power BI

```bash
python src/export_powerbi.py
# Output: powerbi/fraud_dashboard_data.csv
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/model-info` | Model metadata |
| `POST` | `/predict` | Single fraud prediction |
| `POST` | `/predict/explain` | Prediction + **AI explanation** |
| `POST` | `/predict/batch` | Batch predictions |
| `GET` | `/history` | Prediction history from database |

### Example: Prediction with Explanation

```python
import requests

response = requests.post(
    "http://localhost:8000/predict/explain",
    json={
        "V1": -1.35, "V2": -0.07, "V3": 2.53,
        # ... all V1-V28 features ...
        "Amount_log": 5.2, "Hour": 14.0
    }
)

result = response.json()
print(result["ai_explanation"])
```

---

## 📊 Dashboard Pages

1. **🏠 Overview** — KPI metrics, risk distribution, recent alerts, batch analysis
2. **🔍 Transaction Analyzer** — Analyze individual transactions with AI explanations
3. **📊 Model Performance** — ROC, PR-AUC, confusion matrix, SHAP plots
4. **📋 Prediction History** — Query database, filter by risk, export to CSV

---

## 🧠 How the Explainable AI Works

```
Transaction Features
        │
        ▼
┌─────────────────┐
│   XGBoost Model │ ──► Fraud Probability (0.87)
└─────────────────┘
        │
        ▼
┌─────────────────┐
│  SHAP Explainer │ ──► V14: -0.52, V10: -0.38, V17: -0.21
└─────────────────┘
        │
        ▼
┌─────────────────┐
│  LangChain +    │ ──► "This transaction is flagged as HIGH
│  Google Gemini  │      RISK. The primary indicators are
└─────────────────┘      unusual patterns in transaction
                         characteristics, particularly in
                         features associated with known
                         fraud patterns..."
```

---

## 📝 Interview Talking Points

1. **"Why XGBoost over Deep Learning?"** — Tabular data, interpretability, SHAP compatibility
2. **"How do you handle class imbalance?"** — SMOTE, undersampling, cost-sensitive learning
3. **"Why PR-AUC over accuracy?"** — 0.17% fraud rate makes accuracy meaningless
4. **"What is SHAP?"** — Feature attribution based on Shapley values from game theory
5. **"Why LangChain?"** — Converts technical SHAP output to business-friendly explanations
6. **"How is this production-ready?"** — FastAPI, SQLite audit trail, Docker, threshold optimization

---

## 📜 License

This project is for educational and portfolio purposes.

---

## 🙏 Acknowledgments

- [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- [SHAP Library](https://github.com/slundberg/shap)
- [LangChain](https://www.langchain.com/)
- [Google Gemini](https://ai.google.dev/)
- [Streamlit](https://streamlit.io/)
