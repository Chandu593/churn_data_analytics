# 📉 Customer Churn Analytics — End-to-End Data Analytics Pipeline

> *Transforming raw telecom subscriber data into actionable retention intelligence through descriptive, predictive, and prescriptive analytics.*

---

## 📌 Project Overview

Customer churn is one of the most costly challenges in the telecommunications industry. Acquiring a new customer costs 5–10× more than retaining an existing one, making early churn detection a high-value analytics problem.

This project presents a **complete data analytics pipeline** — from raw data ingestion to quantified business ROI — structured across the three pillars of modern analytics:

| Analytics Layer | What It Answers | Techniques Used |
|---|---|---|
| **Descriptive** | What is happening? | EDA, Profiling, Statistical Summary, Visualization |
| **Predictive** | What will happen? | Multi-model comparison, SMOTE, XGBoost Tuning |
| **Prescriptive** | What should we do? | SHAP-based insights, Retention strategy mapping, ROI simulation |

---

## 🗂️ Dataset
- **Source:** `churn-bigml-80.csv` — Telecom subscriber dataset(https://www.kaggle.com/datasets/pratikshapagar2216/churn-bigml-80)
- **Target Variable:** `Churn` (Binary: Churned / Not Churned)
- **Key Features Include:**
  - Call usage metrics (Day, Evening, Night, International minutes & charges)
  - Subscription attributes (International plan, Voice mail plan)
  - Engagement signal: Customer service calls
  - Geographic identifiers: State, Area code
  - Engineered features: Total usage minutes, Total charges, Service call ratio

---

## 🔬 Analytics Pipeline

### 1. Descriptive Analytics

The descriptive phase establishes a comprehensive understanding of the subscriber base before any modeling begins.

**Automated Profiling**
- Used `ydata-profiling` to generate a full statistical profile report — including distributions, correlations, missing value analysis, and data quality flags — in a single pass.

**Visual Storytelling**
- **Churn vs. International Plan:** Subscribers with an international plan exhibit a disproportionately higher churn rate, suggesting unmet expectations around international services.
- **Churn vs. Customer Service Calls:** Box plot analysis reveals churned customers have significantly more service interactions — a strong behavioral signal of dissatisfaction accumulation.

**Conditional Churn Profiling**
Computed the percentage of actual churners satisfying key behavioral conditions:
- Holding an international plan
- Making 4+ customer service calls
- Lacking a voicemail plan
- Falling in the top quartile for daytime usage
- Falling in the top quartile for total charges

This transforms raw distributions into targeted risk profiles.

---

### 2. Data Preprocessing & Feature Engineering

**Preprocessing Steps:**
- Null value audit (zero missing values confirmed)
- Label encoding for binary categoricals: `International plan`, `Voice mail plan`, `Churn`
- One-hot encoding for high-cardinality features: `State`, `Area code`
- Stratified train-test split (80/20) to preserve class distribution
- StandardScaler applied to all continuous numeric features

**Engineered Features:**

| Feature | Formula | Business Rationale |
|---|---|---|
| `Total_usage_minutes` | Day + Eve + Night + Intl minutes | Captures overall consumption behavior |
| `Total_charges` | Day + Eve + Night + Intl charges | Aggregated financial exposure of the subscriber |
| `Service_call_ratio` | Service calls / Account length | Normalizes support burden relative to tenure |

**Class Imbalance Handling — SMOTE**
- The raw dataset is inherently imbalanced (churn is a minority event).
- Applied **Synthetic Minority Over-sampling Technique (SMOTE)** on the training set only — preserving the integrity of the test set for realistic evaluation.
- Post-SMOTE training distribution: balanced 50/50 split.

---

### 3. Predictive Analytics — Multi-Model Benchmarking

Eight classification algorithms were trained and benchmarked to identify the strongest churn signal detector:

| Model | Notes |
|---|---|
| Logistic Regression | Linear baseline |
| Decision Tree | Interpretable, non-linear |
| K-Nearest Neighbors | Instance-based learning |
| Random Forest | Ensemble bagging |
| Gradient Boosting | Sequential ensemble |
| XGBoost | Regularized gradient boosting |
| LightGBM | Leaf-wise boosting, class-weighted |
| SVM | Margin-based classifier |

**Evaluation Philosophy:**
> In churn analytics, **Recall** is the north star metric — a missed churner (False Negative) is revenue lost forever. Precision governs campaign efficiency. Both matter; Recall is prioritized.

Models were evaluated on: Accuracy, Precision, Recall, F1 Score, and ROC-AUC.

**Best Performer: XGBoost** — selected for hyperparameter optimization based on highest Recall.

---

### 4. Model Optimization — Hyperparameter Tuning

`RandomizedSearchCV` with 5-fold cross-validation was applied to XGBoost, searching across:

```
n_estimators, learning_rate, max_depth, min_child_weight,
gamma, subsample, colsample_bytree, reg_alpha, reg_lambda
```

- **Iterations:** 40 random configurations
- **Scoring:** ROC-AUC
- Optimized model re-evaluated on the held-out test set for unbiased performance reporting.

---

### 5. Explainable AI — SHAP Analysis

Model performance alone is insufficient for business adoption. Decision-makers require *why* — not just *what*.

**SHAP (SHapley Additive exPlanations)** was applied using a `TreeExplainer` on the tuned XGBoost model.

**SHAP Summary Plot Insights:**

| Feature | Direction | Interpretation |
|---|---|---|
| `Customer service calls` | High value → High churn risk | Repeated service contacts signal unresolved dissatisfaction |
| `Total_charges` | High value → High churn risk | Heavy billing drives price sensitivity and exits |
| `International plan` | Active plan → Higher churn risk | International plan subscribers have unmet service expectations |
| `Total intl calls` | High value → Lower churn risk | Engaged international users are more sticky |

SHAP bridges the gap between model internals and human-understandable business logic.

---

### 6. Prescriptive Analytics — From Insight to Action

SHAP-derived churn drivers are directly translated into retention playbooks:

**High Customer Service Calls → Proactive Intervention**
Flag subscribers with 3+ service interactions within a billing cycle. Route them to a dedicated retention team before the next bill cycle. Investigate systemic issues causing repeat contacts.

**International Plan Subscribers → Plan Optimization Outreach**
Monitor international usage for sudden drops — a leading indicator of plan abandonment. Trigger personalized offers for upgraded international bundles.

**High Charges → Usage Transparency & Alerts**
Deliver proactive billing summaries and personalized plan recommendations to heavy-usage subscribers before bill shock occurs.

---

### 7. Business ROI Simulation

The final module quantifies the financial impact of deploying the model in a real retention campaign.

**Assumptions:**

| Parameter | Value |
|---|---|
| ARPU (Average Revenue Per User) | ₹2,000/month |
| Retention campaign period | 12 months |
| Retention success rate | 40% |
| Campaign cost per targeted customer | ₹5,000 |

**Derived Metrics:**
- **Predicted churners** = TP + FP (all customers the model flags)
- **Customers saved** = TP × retention success rate
- **Revenue saved** = Customers saved × ARPU × 12
- **Net profit** = Revenue saved − Campaign cost
- **ROI** = (Net profit / Campaign cost) × 100%

This closes the analytics loop — model performance is expressed in business language that stakeholders can act on.

---

## 📊 Key Results Summary

- **Best Model:** XGBoost (tuned via RandomizedSearchCV)
- **Priority Metric:** Recall — maximizing identification of true churners
- **Top Churn Drivers (SHAP):** Customer service calls, Total charges, International plan status
- **Business Output:** Positive ROI simulation demonstrating the financial viability of a model-driven retention campaign

---

## 🗺️ Project Structure

```
customer-churn-analytics/
│
├── DA_Lab.ipynb              # Main analytics notebook
├── churn-bigml-80.csv        # Dataset (add manually)
└── README.md                 # Project documentation
```

---

## 🔮 Potential Extensions

- **Dashboard:** Deploy SHAP insights and churn risk scores to a Power BI / Streamlit dashboard for business users
- **Cohort Analysis:** Segment churn patterns by state, tenure band, and plan type
- **Survival Analysis:** Model time-to-churn using Kaplan-Meier or Cox proportional hazards
- **Real-time Scoring API:** Wrap the tuned model in a Flask/FastAPI service for production scoring pipelines

---

> *Built with a data analytics mindset — not just to predict churn, but to understand it, explain it, and act on it.*
