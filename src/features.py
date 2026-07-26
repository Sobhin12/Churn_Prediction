"""Feature encoding, splitting and scaling for the classification pipeline."""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

TARGET_COL = "Exited"

FEATURES = [
    "CustomerId",
    "CreditScore",
    "Gender",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
    "Geography_Germany",
    "Geography_Spain",
]


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """Label-encode Gender and one-hot encode Geography. Does not scale."""
    df = df.copy()
    df["Gender"] = LabelEncoder().fit_transform(df["Gender"])
    df = pd.get_dummies(df, columns=["Geography"], drop_first=True)
    return df


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = df[FEATURES]
    y = df[TARGET_COL]
    return X, y


def train_test_split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler
