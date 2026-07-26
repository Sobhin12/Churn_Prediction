"""End-to-end runnable pipeline: `python -m src.pipeline [--data-path PATH]`."""

import argparse
from pathlib import Path

from . import classifiers, data, features, survival


def run_classification(csv_path: Path) -> dict[str, classifiers.EvalResult]:
    df = data.load_data(csv_path)
    df_encoded = features.encode_features(df)
    X, y = features.split_features_target(df_encoded)
    X_train, X_test, y_train, y_test = features.train_test_split_data(X, y)
    X_train_scaled, X_test_scaled, _ = features.scale_features(X_train, X_test)

    trainers = {
        "RandomForest": classifiers.train_random_forest,
        "LogisticRegression": classifiers.train_logistic_regression,
        "SVM": classifiers.train_svm,
        "KNN": classifiers.train_knn,
        "GBM": classifiers.train_gbm,
    }

    results = {}
    for name, trainer in trainers.items():
        model = trainer(X_train_scaled, y_train)
        results[name] = classifiers.evaluate(name, y_test, model.predict(X_test_scaled))

    for name, trainer in {
        "XGBoost": classifiers.train_xgboost,
        "CatBoost": classifiers.train_catboost,
        "LightGBM": classifiers.train_lightgbm,
    }.items():
        model, _best_params = trainer(X_train_scaled, y_train)
        results[name] = classifiers.evaluate(name, y_test, model.predict(X_test_scaled))

    return results


def run_survival(csv_path: Path):
    df = data.load_data(csv_path)
    df_encoded = features.encode_features(df)
    X, y = features.split_features_target(df_encoded)

    return survival.fit_cox_model(X, y)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate churn classification models.")
    parser.add_argument("--data-path", type=Path, default=data.DEFAULT_DATA_PATH)
    args = parser.parse_args()

    results = run_classification(args.data_path)
    for name, result in results.items():
        print(f"{name}: accuracy={result.accuracy:.4f}")


if __name__ == "__main__":
    main()
