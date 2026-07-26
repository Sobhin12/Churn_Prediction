"""Time-to-churn survival model.

`duration_col` must be in real, unscaled units (e.g. years of Tenure)
since a negative duration is meaningless. Always pass the unscaled,
encoded feature frame here -- never the StandardScaler output.
"""

import pandas as pd
from lifelines import CoxPHFitter


def fit_cox_model(
    X: pd.DataFrame,
    y: pd.Series,
    duration_col: str = "Tenure",
    event_col: str = "Exited",
) -> CoxPHFitter:
    df = X.copy()
    df[event_col] = y.values

    cph = CoxPHFitter()
    cph.fit(df, duration_col=duration_col, event_col=event_col)
    return cph
