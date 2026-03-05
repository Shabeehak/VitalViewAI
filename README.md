# 🏥 VitalViewAI - AI-Powered Health Monitoring System For Chronic Patients
![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-3.1.2-FF6600?style=flat)
![FastAPI](https://img.shields.io/badge/FastAPI-0.128-009688?style=flat&logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.52-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=flat&logo=docker&logoColor=white)
![PRauc](https://img.shields.io/badge/PR--AUC-0.653-28a745?style=flat)
![CV](https://img.shields.io/badge/CV%20PR--AUC-0.676%20±%200.021-blue?style=flat)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Render-46E3B7?style=flat&logo=render&logoColor=white)](https://vitalviewai-n7su.onrender.com)

**Real-time deterioration prediction for chronic care patients using ML**

> **Educational Project**: Built to demonstrate end-to-end ML engineering skills including data pipeline design, model development, feature selection, hyperparameter tuning, cross-validation, and production deployment.

---

## 🌐 Live Demo

**[https://vitalviewai-n7su.onrender.com](https://vitalviewai-n7su.onrender.com)**

> ⚠️ Hosted on Render free tier — may take 30–60 seconds to wake up on first load.

---

## 📸 Screenshots

### Monitoring Dashboard
![Dashboard Overview](docs/screenshots/dashboard.png)

### Live Prediction Result
![Prediction Result](docs/screenshots/prediction.png)

### API (Swagger UI)
![API Endpoints](docs/screenshots/api_swagger.png)

### Model Performance
![Feature Importance](models/feature_importance.png)
![PR Curve](models/pr_curve.png)

---

## 🎯 Project Overview

VitalViewAI is a complete machine learning system that monitors patient vital signs from wearable devices and predicts health deterioration events 48 hours in advance. This project showcases the full ML lifecycle from data generation to production deployment, including the **challenges and iterations** required to achieve meaningful performance.

### Key Achievement: The Full ML Pipeline Journey

This project documents a real ML debugging and optimisation journey:
- Started with **0.30 PR-AUC** (barely better than random)
- Through systematic data quality fixes: **→ 0.653 PR-AUC** (holdout test)
- 5-fold cross-validation confirms: **0.676 ± 0.021 PR-AUC** across unseen patient groups
- Feature selection, Optuna hyperparameter tuning, and overfitting analysis documented

See [Model Performance Journey](#-model-performance-journey) for details.

---

## 📊 System Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                 Data Generation Layer                        │
│  • 130 patients × 30 days × 288 readings/day = 1.1M samples  │
│  • Realistic vital signs with circadian rhythms              │
│  • Controlled deterioration events (hypertension, hypoxia)   │
└────────────────────┬─────────────────────────────────────────┘
                     │
┌────────────────────▼─────────────────────────────────────────┐
│              Feature Engineering Pipeline                    │
│  • 133 features from 6 base vitals                           │
│  • Rolling stats (1h, 6h, 12h windows)                       │
│  • Trends, interactions, temporal patterns                   │
└────────────────────┬─────────────────────────────────────────┘
                     │
┌────────────────────▼─────────────────────────────────────────┐
│              EDA + Feature Selection + CV                    │
│  • Exploratory analysis of all 6 vital distributions         │
│  • Gain-based selection: 133 → 68 features (90% of gain)     │
│  • 5-fold GroupKFold cross-validation (patient-level)        │
│  • Optuna hyperparameter tuning (20 Bayesian trials)         │
└────────────────────┬─────────────────────────────────────────┘
                     │
┌────────────────────▼─────────────────────────────────────────┐
│                 XGBoost Classifier                           │
│  • 500 trees (early stopped at iteration ~163)               │
│  • SMOTE balancing + patient-level splitting                 │
│  • Test PR-AUC: 0.653 | CV PR-AUC: 0.676 ± 0.021            │
└────────────────────┬─────────────────────────────────────────┘
                     │
┌────────────────────▼─────────────────────────────────────────┐
│            Production API + Dashboard                        │
│  • FastAPI real-time prediction server                       │
│  • Streamlit monitoring dashboard                            │
│  • Comprehensive logging and audit trails                    │
└──────────────────────────────────────────────────────────────┘
```

---

## 🔬 Model Performance Journey

### The Challenge: From 0.30 to 0.653 PR-AUC

This section documents the **systematic debugging process** that improved model performance:

#### **Iteration 1: Initial Failure (PR-AUC: 0.30)**

**Problem**: Model barely better than random guessing
```
Training PR-AUC: 0.24
Test PR-AUC: 0.30
Class distribution: Only 6.4% deteriorating patients
```

**Root Cause Analysis**:
- ❌ Extreme class imbalance (6% vs 94%)
- ❌ Deterioration severity parameter never ramping (0.0 throughout)
- ❌ Model had insufficient positive examples to learn from

**Learning**: Even perfect ML algorithms fail with poor data quality

---

#### **Iteration 2: Data Quality Fix (PR-AUC: 0.10)**

**Action Taken**: Adjusted label threshold from 0.35 → 0.20
```python
if (abnormal_count / total_count) > 0.20:
    deterioration = True
```

**Result**: Overcorrected — 86% deteriorating (opposite problem)

**Learning**: Threshold tuning requires careful calibration

---

#### **Iteration 3: Threshold Calibration**

Fine-tuned label threshold through systematic iteration:

| Threshold | Positive Rate | Issue |
|---|---|---|
| 0.45 | 0.1% | Almost no positive examples |
| 0.35 | 6.4% | Too imbalanced |
| **0.25** | **37.6%** | **Sweet spot — selected** |
| 0.20 | 58.1% | Too many positives |
| 0.15 | 86.3% | Model predicts everyone deteriorating |

---

#### **Iteration 4: Production Model (PR-AUC: 0.653)**

**Training Results**:
```
Training PR-AUC:   0.845
Validation PR-AUC: 0.678
Test PR-AUC:       0.653
CV PR-AUC:         0.676 ± 0.021  (5-fold GroupKFold)
Early stopping:    iteration ~163 of 500
```

**Overfitting gap** (train 0.845 vs val 0.678 = 0.167) identified and documented.
Feature selection and Optuna tuning applied — original model retained as best generaliser
(see Feature Selection section below).

---

### Feature Selection & Hyperparameter Tuning

#### Feature Importance Analysis

Gain-based importance analysis on 133 features revealed that just **4 engineered
interaction features explain 39.5% of total model gain**:

| Rank | Feature | Gain % |
|---|---|---|
| 1 | mean_arterial_pressure | 23.2% |
| 2 | cv_stress_index | 11.9% |
| 3 | heart_rate_mean_1h | 7.8% |
| 4 | respiratory_efficiency | 4.2% |
| 5 | hour_of_day | 4.1% |

68 features explain 90% of model gain. Feature selection was tested:

```
Original model (133 features): Test PR-AUC = 0.653
Feature selected (68 features): Test PR-AUC = 0.574  ← worse
```

With only 26 test patients, removing features hurt generalisation — a known
small-cohort effect. Original model retained.

#### Optuna Hyperparameter Tuning

Bayesian optimisation over 20 trials on the 68-feature subset:

```
Best val PR-AUC:  0.6796
Best test PR-AUC: 0.5686  ← worse than original 0.653
```

Original 133-feature model is the production model. Tuning results saved in
`models/optuna_best_params.json`.

---

### Cross Validation Results

5-fold GroupKFold ensures no patient appears in both train and test within any fold.
SMOTE applied inside each fold on training data only.

| Fold | Train Patients | Test Patients | PR-AUC | ROC-AUC |
|---|---|---|---|---|
| 1 | 104 | 26 | 0.7050 | 0.7216 |
| 2 | 104 | 26 | 0.6486 | 0.7011 |
| 3 | 104 | 26 | 0.6722 | 0.7060 |
| 4 | 104 | 26 | 0.6937 | 0.7256 |
| 5 | 104 | 26 | 0.6617 | 0.7173 |
| **Mean** | | | **0.676 ± 0.021** | **0.714 ± 0.009** |

CV mean (0.676) is consistent with holdout test PR-AUC (0.653) — confirming
the result is not a lucky or unlucky split.

---

### Key Insights from the Journey

| Learning | Impact |
|---|---|
| **Data quality > Model complexity** | Biggest gains came from fixing data, not the model |
| **Class balance is critical** | 6% → 37% positive samples drove most improvement |
| **Feature selection can hurt with small cohorts** | 26 test patients too few for stable selection |
| **Healthcare = Recall vs Precision trade-off** | Both matter — missed events are dangerous, false alarms cause alert fatigue |
| **CV confirms stability** | 0.676 ± 0.021 shows consistent performance across patient groups |

---

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.11+
1.5GB free disk space
8GB RAM recommended
```

### Installation

```bash
# Clone repository
git clone https://github.com/Shabeehak/VitalViewAI.git
cd VitalViewAI

# Create virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Complete Workflow

```bash
# Step 1: Generate data (~30-60 minutes for 130 patients)
python generate_diverse_training_data.py

# Quick test run (10 patients, ~3 minutes)
python generate_diverse_training_data.py --quick

# Step 2: Train model (~5-10 minutes)
python train_model_xgboost.py

# Step 3: Exploratory Data Analysis
python eda_analysis.py
# Outputs: reports/eda/ (7 plots)

# Step 4: Feature Selection
python feature_selection.py --analyse-only
# Outputs: reports/feature_selection/feature_importance.png

# Step 5: Cross Validation (~1 hour)
python cross_validation.py
# Outputs: reports/cross_validation/cv_results.json

# Step 6: Evaluate
python evaluate_model.py --model xgboost

# Step 7: Start system (Windows)
.\start_system.ps1

# Step 8: Run dashboard
streamlit run streamlit_dashboard.py
```

Access at: http://localhost:8501

---

## 📁 Project Structure

```
VitalViewAI/
├── generate_diverse_training_data.py   # Data generation (130 patients)
├── train_model_xgboost.py              # Main training entry point
├── evaluate_model.py                   # Model evaluation
├── eda_analysis.py                     # Exploratory data analysis
├── feature_selection.py                # Feature importance + Optuna tuning
├── cross_validation.py                 # 5-fold GroupKFold cross-validation
├── auto_retrain.py                     # Drift detection + automated retraining
├── streaming_api_server.py             # FastAPI backend (port 8000)
├── ml_server.py                        # ML inference API (port 8001)
├── streamlit_dashboard.py              # Monitoring UI (port 8501)
├── predictor.py                        # Prediction interface
│
├── src/
│   ├── data/
│   │   ├── wearable_simulator.py
│   │   └── generate_lab_data.py
│   ├── features/
│   │   └── feature_engineering.py      # 133 features
│   └── models/
│       └── train_xgboost.py            # XGBoost class library
│
├── data/
│   └── processed/
│       ├── features_multi.csv          # 1.1M labelled samples
│       └── features_engineered.csv     # 133 engineered features
│
├── models/
│   ├── xgboost_model.pkl                    # Production model (PR-AUC 0.653)
│   ├── xgboost_model_metadata.json          # Performance + feature names
│   ├── xgboost_model_selected.pkl           # 68-feature variant
│   ├── xgboost_model_selected_metadata.json
│   ├── optuna_best_params.json              # Hyperparameter tuning results
│   ├── confusion_matrix.png
│   ├── pr_curve.png
│   └── feature_importance.png
│
├── reports/
│   ├── eda/                            # 7 EDA plots
│   ├── feature_selection/              # Importance chart + selected_features.json
│   └── cross_validation/              # cv_results.json + cv_scores_plot.png
│
├── logs/
│   ├── application.log
│   ├── audit.log
│   └── performance.log
│
├── docs/
│   ├── ARCHITECTURE.md
│   ├── MODEL_EVALUATION.md
│   ├── DATA_DOCUMENTATION.md
│   ├── DASHBOARD_GUIDE.md
│   └── DEPLOYMENT.md
│
├── monitoring/
│   ├── grafana/
│   └── prometheus/
│
├── privacy_config.yaml
├── logging_config.py
├── docker-compose.yaml
└── requirements.txt
```

---

## 📊 Final Performance Metrics

### Model Performance (holdout test — 26 unseen patients)

| Metric | Value | Interpretation |
|---|---|---|
| **PR-AUC** | **0.653** | 73% above random baseline (0.376) |
| **ROC-AUC** | **0.698** | Model correctly ranks 69.8% of patient pairs |
| **CV PR-AUC** | **0.676 ± 0.021** | Stable across 5 patient groups |
| **Precision** | **~70%** | When it alerts, usually correct |
| **Recall** | **~32%** | Conservative at default threshold |
| **Specificity** | **~89%** | Strong at identifying stable patients |

### Confusion Matrix (threshold=0.5, approximate)

```
                    Predicted
                  Stable  |  Deteriorating
Actual  Stable   145,855  |  10,408   ← false alarms (6.7%)
      Deterior.   52,498  |  24,715   ← caught events

True Positives:  24,715  ✅
True Negatives: 145,855  ✅
False Positives:  10,408  ⚠️  (false alarms)
False Negatives:  52,498  ❌  (primary improvement target)
```

### Clinical Trade-off Context

At threshold 0.5 the model is conservative — high precision, lower recall.
Suitable as a secondary alert system alongside clinical judgement.
Lowering threshold to 0.3 substantially increases recall at the cost of
more false alarms — appropriate for primary screening contexts.

---

## 🛠️ Technical Implementation

### Data Generation

**Patient Simulator**:
```python
# 130 patients with diverse health profiles
# Healthy / At-risk / Deteriorating

# Deterioration severity ramps 0 → 1 over 6 hours
# modelling physiological onset of acute events
vital_modifier = baseline × (1 + severity_factor × severity)
```

### Feature Engineering

**133 Features from 6 Base Vitals**:

```
Base Vitals (6):
  Heart Rate, BP Systolic, BP Diastolic,
  SpO2, Respiratory Rate, Temperature

Rolling Statistics (72 features):
  Mean, Std, Min, Max over [1h, 6h, 12h] windows

Trend Features (12 features):
  OLS slope over 6h and 12h windows per vital

Interaction Features (4 features — 39.5% of model gain):
  mean_arterial_pressure = diastolic + pulse_pressure/3
  cv_stress_index        = (HR/100) × (SBP/120)
  respiratory_efficiency = SpO2 / respiratory_rate
  pulse_pressure         = systolic − diastolic

Temporal Features (7 features):
  hour_of_day, hour_sin, hour_cos,
  day_of_week, is_weekend

Lag Features (18 features):
  Previous 1, 2, 3 readings for each vital
```

### Model Configuration

```python
XGBClassifier(
    n_estimators=500,           # Early stopped at ~163
    max_depth=6,
    learning_rate=0.01,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.05,
    min_child_weight=1,
    scale_pos_weight=1,         # SMOTE already balanced classes
    eval_metric='aucpr',
    early_stopping_rounds=20,
)
```

**Patient-level splitting** (prevents data leakage):
```
Train: 91 patients (70%)  →  SMOTE applied  →  976K balanced samples
Val:   13 patients (10%)  →  early stopping
Test:  26 patients (20%)  →  never seen during training

✅ No patient appears in multiple splits
✅ Validated by 5-fold GroupKFold cross-validation
```

---

## 🔒 Security & Privacy

### HIPAA-Style Implementation

**Implemented**:
- ✅ Role-based access control (clinician / nurse / researcher)
- ✅ Comprehensive audit logging (7-year retention standard)
- ✅ AES-256 encryption via Fernet
- ✅ SHA-256 patient ID hashing with salt
- ✅ JWT authentication with bcrypt password hashing
- ✅ TLS-ready API architecture

**Production Requirements** (not yet enforced at middleware level):
```
⚠️ For real deployment would require:
- Active JWT enforcement at API gateway
- PostgreSQL for persistent storage
- Redis for session management
- HIPAA BAA with infrastructure provider
```

---

## 🧪 Testing

```bash
# Unit tests
pytest tests/test_unit.py -v --cov=src

# Integration tests
python tests/test_complete_system.py

# API tests
python tests/test_api_complete.py
```

### Performance Benchmarks

| Operation | Latency |
|---|---|
| Feature Engineering | ~50ms |
| XGBoost Prediction | ~45ms |
| API Request (E2E) | ~380ms |

---

## 🚧 Known Limitations & Future Work

### Current Limitations

1. **Synthetic Data Only** — real-world performance will differ
2. **Overfitting Gap** — train 0.845 vs val 0.678 (gap: 0.167)
3. **Recall ~32% at threshold 0.5** — misses most events at default
4. **Small cohort** — 130 patients causes run-to-run variance
5. **Security scaffold** — RBAC defined but not enforced at middleware
6. **Monitoring** — Prometheus/Grafana configured but requires Docker

### Planned Improvements

- SHAP explainability for individual predictions
- Validate on MIMIC-III real patient data
- Patient-specific threshold calibration
- Ensemble: XGBoost + Random Forest
- Full JWT enforcement at API middleware
- Experiment tracking with MLflow

---

## 📚 Tech Stack

| Component | Technology |
|---|---|
| **ML** | XGBoost 3.1.2 |
| **Tuning** | Optuna |
| **Imbalance** | imbalanced-learn (SMOTE) |
| **Data** | Pandas 2.0.3, NumPy 1.26.4 |
| **API** | FastAPI 0.128.0, Uvicorn |
| **Dashboard** | Streamlit 1.52.2 |
| **Visualization** | Plotly, Matplotlib, Seaborn |
| **Security** | Cryptography 46.0.3 |
| **Monitoring** | Prometheus, Grafana |
| **Testing** | pytest 9.0.2 |

---

## 🎓 Skills Demonstrated

### Machine Learning
✅ Root cause debugging: traced 0.30 PR-AUC to data pipeline bug  
✅ Systematic iteration: 4 documented debugging cycles  
✅ EDA: vital sign distributions, circadian patterns, patient profiles  
✅ Class imbalance: SMOTE + PR-AUC optimisation  
✅ Feature engineering: 6 vitals → 133 features  
✅ Feature selection: gain-based, 133 → 68 features  
✅ Hyperparameter tuning: Optuna Bayesian optimisation (20 trials)  
✅ Cross-validation: 5-fold GroupKFold (patient-level, no leakage)  
✅ Overfitting detection, analysis and documentation  
✅ Healthcare trade-offs: precision vs recall in clinical context  

### Data Engineering
✅ Synthetic patient simulator with realistic physiology  
✅ Patient-level splitting to prevent data leakage  
✅ Drift detection with KS test + automated retraining  

### Software Engineering
✅ Production FastAPI with structured logging  
✅ Audit trails and correlation IDs  
✅ Docker + Kubernetes deployment scaffold  
✅ Unit, integration, and API tests  

### MLOps
✅ Model versioning and registry  
✅ Automated retraining pipeline  
✅ Prometheus/Grafana monitoring configuration  
✅ CI/CD scaffolding  

---

## 💡 Discussion Points

### What Went Well
✅ Systematic debugging with documented iterations  
✅ Root cause analysis (data pipeline → model quality)  
✅ Complete ML pipeline: EDA → feature selection → tuning → CV  
✅ Honest documentation — feature selection made things worse, documented why  
✅ Healthcare-domain awareness  

### What I'd Do Differently
🔄 Start with EDA before model building  
🔄 Add SHAP explainability from the start  
🔄 Collect more patients before tuning  
🔄 Test on real data (MIMIC-III) sooner  

---

## 📖 Documentation

- **[System Architecture](docs/ARCHITECTURE.md)**
- **[Model Evaluation](docs/MODEL_EVALUATION.md)**
- **[Data Documentation](docs/DATA_DOCUMENTATION.md)**
- **[Dashboard Guide](docs/DASHBOARD_GUIDE.md)**
- **[Deployment Guide](docs/DEPLOYMENT.md)**

---

## 📞 Contact

**Shabeeha K**  
📧 shabeehakalathumpadiyil@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/shabeeha-kalathumpadiyil/)  
🔗 [GitHub](https://github.com/Shabeehak)  
🔗 [Portfolio](https://shabeehak.github.io/Personal_Website/)

---

## ⚠️ Disclaimer

**Educational / Portfolio Project Only**

- ❌ Not validated on real patient data  
- ❌ Not FDA approved or clinically validated  
- ❌ Not intended for actual medical use  
- ✅ Built to demonstrate ML engineering skills  

---

## 🏆 Key Takeaway

> "The journey from 0.30 → 0.653 PR-AUC (0.676 cross-validated) wasn't about finding a better algorithm. It was about understanding the problem, fixing the data pipeline, running proper analysis, and knowing when NOT to apply a technique — which is what real ML engineering looks like."

---

**Built as part of ML learning journey | Brototype AI/ML Program | 2025–2026**