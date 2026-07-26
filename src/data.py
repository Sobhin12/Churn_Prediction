"""Loading and cleaning of the raw churn dataset."""

from pathlib import Path

import pandas as pd

DEFAULT_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "Customer-Churn-Records.csv"

TARGET_COL = "Exited"


def load_data(path: Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.drop(columns=["RowNumber"])
    return df
