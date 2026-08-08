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

## 📁 Project Structure

```
src/
  data.py         # load & clean the raw CSV
  features.py     # encoding, train/test split, scaling
  classifiers.py   # train/evaluate each classification model
  explain.py       # SHAP explainability
  survival.py      # Cox Proportional Hazards model
  pipeline.py      # runnable end-to-end script
analysis.ipynb      # EDA and reporting notebook; imports from src/
data/               # place Customer-Churn-Records.csv here (git-ignored)
requirements.txt    # all dependencies
```

Reusable logic lives in `src/`; `analysis.ipynb` is the exploratory/reporting
layer that calls into it. Run the classifiers end-to-end with:

```
python -m src.pipeline --data-path data/Customer-Churn-Records.csv
```

**Note:** the Cox model uses `Tenure` as the survival duration — this must
stay in unscaled, real-world units (years), not the standardized features
used for classification.
