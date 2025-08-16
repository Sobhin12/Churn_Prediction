# Customer Churn Prediction with ML & Survival Analysis

This project focuses on predicting **customer churn** using machine learning and survival analysis techniques. The goal was not only to classify churners but also to **estimate the time-to-churn** and explain the driving factors behind customer attrition.

## Key Highlights

* ✅ **Achieved 86.7% accuracy** using **LightGBM**, outperforming XGBoost and CatBoost.
* 🔍 **Engineered 15+ behavioral & transactional features** to improve signal capture for churn prediction.
* ⏳ Built a **Cox Proportional Hazards Survival Model** to estimate **time-to-churn**, achieving a **concordance index of 0.72**.
* 📊 Applied **SHAP explainability framework** to interpret model outputs and **identify key churn drivers**.

## 🛠️ Tech Stack

* **Languages**: Python
* **ML Models**: XGBoost, LightGBM, CatBoost
* **Survival Analysis**: Cox Proportional Hazards Model
* **Explainability**: SHAP
* **Libraries**: pandas, numpy, scikit-learn, lifelines, matplotlib, seaborn


## 📊 Results

* **Classification Models**

  * LightGBM → **86.7% accuracy**
  * XGBoost & CatBoost → slightly lower performance

* **Survival Analysis**

  * Cox Proportional Hazards model → **Concordance Index: 0.72**
  * Identified risk factors contributing to **earlier churn probability**

* **Explainability**

  * SHAP analysis revealed top churn drivers (e.g., reduced engagement frequency, lower transactional activity, delayed payments).


## 🔮 Future Work

* Incorporate **deep learning survival models** (DeepSurv, RNN-based time-to-event models).
* Extend explainability with **counterfactual churn analysis**.
* Deploy as a **churn prediction API** for real-time business applications.
