import argparse
import itertools
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import joblib

DATA_PATH = "car_sales_data.csv"
MODEL_PATH = "car_brand_model.joblib"
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"


def load_data(path=DATA_PATH):
    return pd.read_csv(path)


def build_model(n_estimators=200, max_depth=None, min_samples_split=2,
                min_samples_leaf=1, max_features="sqrt"):
    categorical_features = ["car_type", "month", "year"]
    numeric_features = ["price", "mileage"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ],
        remainder="passthrough",
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    max_features=max_features,
                    random_state=42,
                    class_weight="balanced",
                    n_jobs=-1,
                ),
            ),
        ]
    )
    return pipeline


def train(n_estimators=200, max_depth=None, min_samples_split=2,
          min_samples_leaf=1, max_features="sqrt", experiment_name="car_brand_model"):
    df = load_data()
    X = df[["price", "mileage", "car_type", "month", "year"]]
    y = df["brand"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("min_samples_split", min_samples_split)
        mlflow.log_param("min_samples_leaf", min_samples_leaf)
        mlflow.log_param("max_features", max_features)

        model = build_model(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
        )
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions, zero_division=0, output_dict=True)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision_macro", report["macro avg"]["precision"])
        mlflow.log_metric("recall_macro", report["macro avg"]["recall"])
        mlflow.log_metric("f1_score_macro", report["macro avg"]["f1-score"])

        mlflow.sklearn.log_model(model, "model")

        print(f"\nRun ID: {mlflow.active_run().info.run_id}")
        print(f"Accuracy: {accuracy:.4f}")
        print(classification_report(y_test, predictions, zero_division=0))

        joblib.dump(model, MODEL_PATH)
        print(f"Model saved to {MODEL_PATH}")


def tune():
    param_grid = {
        "n_estimators": [50, 100, 200, 500],
        "max_depth": [10, 20, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
    }

    keys = list(param_grid.keys())
    total = 1
    for v in param_grid.values():
        total *= len(v)

    print(f"Starting hyperparameter tuning — {total} combinations\n")

    for i, values in enumerate(itertools.product(*param_grid.values()), 1):
        params = dict(zip(keys, values))
        print(f"[{i}/{total}] Running with {params}")
        train(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_split=params["min_samples_split"],
            min_samples_leaf=params["min_samples_leaf"],
            max_features=params["max_features"],
            experiment_name="car_brand_tuning",
        )
        print("-" * 60)

    print(f"\nTuning complete! Run 'mlflow ui' to view results.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train car brand model with MLflow tracking")
    parser.add_argument("--tune", action="store_true", help="Run hyperparameter tuning")
    args = parser.parse_args()

    if args.tune:
        tune()
    else:
        train()
