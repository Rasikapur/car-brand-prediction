import argparse
import itertools
import os
import time
import tempfile

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import joblib

DATA_PATH = "car_sales_data.csv"
MODEL_PATH = "car_brand_model.joblib"
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")


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


def plot_confusion_matrix(y_true, y_pred, class_names, output_path):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=45)
    plt.tight_layout()
    fig.savefig(output_path, dpi=100)
    plt.close(fig)


def plot_feature_importance(model, feature_names, output_path):
    rf = model.named_steps["classifier"]
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(len(importances)), importances[indices])
    ax.set_xticks(range(len(importances)))
    ax.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha="right")
    ax.set_title("Feature Importance")
    ax.set_ylabel("Importance")
    plt.tight_layout()
    fig.savefig(output_path, dpi=100)
    plt.close(fig)


def plot_class_distribution(y_train, y_test, class_names, output_path):
    train_counts = pd.Series(y_train).value_counts().reindex(class_names, fill_value=0)
    test_counts = pd.Series(y_test).value_counts().reindex(class_names, fill_value=0)

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(class_names))
    width = 0.35
    ax.bar(x - width / 2, train_counts.values, width, label="Train")
    ax.bar(x + width / 2, test_counts.values, width, label="Test")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_title("Class Distribution (Train vs Test)")
    ax.set_ylabel("Count")
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_path, dpi=100)
    plt.close(fig)


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

    with mlflow.start_run() as run:
        run_id = run.info.run_id

        mlflow.set_tag("model_type", "RandomForestClassifier")
        mlflow.set_tag("data_source", os.path.basename(DATA_PATH))
        mlflow.set_tag("num_classes", len(y.unique()))
        mlflow.set_tag("num_samples", len(df))
        mlflow.set_tag("experiment_name", experiment_name)

        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("min_samples_split", min_samples_split)
        mlflow.log_param("min_samples_leaf", min_samples_leaf)
        mlflow.log_param("max_features", max_features)
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("test_samples", len(X_test))

        mlflow.log_metric("n_features", X.shape[1])

        start_time = time.time()
        model = build_model(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
        )
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
        mlflow.log_metric("training_duration_seconds", round(train_time, 2))

        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions, zero_division=0, output_dict=True)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision_macro", report["macro avg"]["precision"])
        mlflow.log_metric("recall_macro", report["macro avg"]["recall"])
        mlflow.log_metric("f1_score_macro", report["macro avg"]["f1-score"])
        mlflow.log_metric("precision_weighted", report["weighted avg"]["precision"])
        mlflow.log_metric("recall_weighted", report["weighted avg"]["recall"])
        mlflow.log_metric("f1_score_weighted", report["weighted avg"]["f1-score"])

        classes = sorted(y.unique())
        for cls in classes:
            cls_key = str(cls)
            if cls_key in report:
                mlflow.log_metric(f"precision_{cls}", report[cls_key]["precision"])
                mlflow.log_metric(f"recall_{cls}", report[cls_key]["recall"])
                mlflow.log_metric(f"f1_score_{cls}", report[cls_key]["f1-score"])
                mlflow.log_metric(f"support_{cls}", report[cls_key]["support"])

        cat_encoder = model.named_steps["preprocessor"].named_transformers_["cat"]
        encoded_categories = cat_encoder.get_feature_names_out()
        numeric_features = ["price", "mileage"]
        all_feature_names = list(encoded_categories) + numeric_features

        with tempfile.TemporaryDirectory() as tmpdir:
            cm_path = os.path.join(tmpdir, "confusion_matrix.png")
            plot_confusion_matrix(y_test, predictions, classes, cm_path)
            mlflow.log_artifact(cm_path, artifact_path="plots")

            fi_path = os.path.join(tmpdir, "feature_importance.png")
            plot_feature_importance(model, all_feature_names, fi_path)
            mlflow.log_artifact(fi_path, artifact_path="plots")

            dist_path = os.path.join(tmpdir, "class_distribution.png")
            plot_class_distribution(y_train, y_test, classes, dist_path)
            mlflow.log_artifact(dist_path, artifact_path="plots")

            report_path = os.path.join(tmpdir, "classification_report.txt")
            with open(report_path, "w") as f:
                f.write(classification_report(y_test, predictions, zero_division=0))
            mlflow.log_artifact(report_path, artifact_path="metrics")

            cm_data_path = os.path.join(tmpdir, "confusion_matrix.csv")
            cm = confusion_matrix(y_test, predictions)
            cm_df = pd.DataFrame(cm, index=classes, columns=classes)
            cm_df.to_csv(cm_data_path)
            mlflow.log_artifact(cm_data_path, artifact_path="metrics")

        input_schema = mlflow.types.Schema([
            mlflow.types.ColSpec(mlflow.types.DataType.double, "price"),
            mlflow.types.ColSpec(mlflow.types.DataType.double, "mileage"),
            mlflow.types.ColSpec(mlflow.types.DataType.string, "car_type"),
            mlflow.types.ColSpec(mlflow.types.DataType.long, "month"),
            mlflow.types.ColSpec(mlflow.types.DataType.long, "year"),
        ])
        output_schema = mlflow.types.Schema([
            mlflow.types.ColSpec(mlflow.types.DataType.string, "predicted_brand"),
        ])
        signature = mlflow.models.ModelSignature(inputs=input_schema, outputs=output_schema)

        mlflow.sklearn.log_model(
            model,
            "model",
            signature=signature,
            registered_model_name="car_brand_predictor" if experiment_name == "car_brand_model" else None,
        )

        print(f"\nRun ID: {run_id}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Training time: {train_time:.2f}s")
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

    print(f"Starting hyperparameter tuning -- {total} combinations\n")

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
