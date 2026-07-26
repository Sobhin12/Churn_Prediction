"""Training and evaluation helpers for the classification models.

Each `train_*` function fits one model on already-scaled features. The
GridSearchCV-based ones (XGBoost, CatBoost, LightGBM) return the best
estimator together with its best hyperparameters.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


@dataclass
class EvalResult:
    model_name: str
    accuracy: float
    confusion_matrix: np.ndarray
    classification_report: str


def evaluate(model_name: str, y_true, y_pred) -> EvalResult:
    return EvalResult(
        model_name=model_name,
        accuracy=accuracy_score(y_true, y_pred),
        confusion_matrix=confusion_matrix(y_true, y_pred),
        classification_report=classification_report(y_true, y_pred),
    )


def train_random_forest(X_train, y_train, n_estimators: int = 100, random_state: int = 42):
    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)
    return model


def train_logistic_regression(X_train, y_train, random_state: int = 42):
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(random_state=random_state)
    model.fit(X_train, y_train)
    return model


def train_svm(X_train, y_train, random_state: int = 42):
    from sklearn.svm import SVC

    model = SVC(kernel="linear", random_state=random_state)
    model.fit(X_train, y_train)
    return model


def train_knn(X_train, y_train, n_neighbors: int = 5):
    from sklearn.neighbors import KNeighborsClassifier

    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)
    return model


def train_gbm(X_train, y_train, n_estimators: int = 100, random_state: int = 42):
    from sklearn.ensemble import GradientBoostingClassifier

    model = GradientBoostingClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train):
    from sklearn.model_selection import GridSearchCV
    from xgboost import XGBClassifier

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.1, 0.2],
        "subsample": [0.8, 1],
        "colsample_bytree": [0.8, 1],
    }
    grid = GridSearchCV(
        XGBClassifier(eval_metric="logloss"),
        param_grid,
        scoring="roc_auc",
        cv=5,
        verbose=1,
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_


def train_catboost(X_train, y_train):
    from catboost import CatBoostClassifier
    from sklearn.model_selection import GridSearchCV

    param_grid = {
        "depth": [4, 6, 8],
        "learning_rate": [0.01, 0.1],
        "iterations": [100, 200],
        "l2_leaf_reg": [1, 3, 5],
    }
    grid = GridSearchCV(
        CatBoostClassifier(verbose=0),
        param_grid,
        scoring="roc_auc",
        cv=5,
        verbose=1,
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_


def train_lightgbm(X_train, y_train):
    import lightgbm as lgb
    from sklearn.model_selection import GridSearchCV

    param_grid = {
        "num_leaves": [31, 50],
        "max_depth": [-1, 10, 20],
        "learning_rate": [0.01, 0.1],
        "n_estimators": [100, 200],
        "subsample": [0.8, 1.0],
    }
    grid = GridSearchCV(
        lgb.LGBMClassifier(),
        param_grid,
        scoring="roc_auc",
        cv=5,
        verbose=1,
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_


def feature_importance(model, feature_names) -> pd.DataFrame:
    return (
        pd.DataFrame({"Feature": feature_names, "Importance": model.feature_importances_})
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )
