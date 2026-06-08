import mlflow

# Point to your MLflow server
mlflow.set_tracking_uri("http://localhost:5000")

# Create or use an experiment
mlflow.set_experiment("test-experiment")

with mlflow.start_run():
    mlflow.log_param("learning_rate", 0.01)
    mlflow.log_param("batch_size", 64)

    mlflow.log_metric("accuracy", 0.95)
    mlflow.log_metric("loss", 0.05)

    print("Run logged successfully")
