# Car Brand Predictor

Flask web app that predicts car brand based on price, mileage, car type, month, and year using a Random Forest classifier.

## Features

- Predict car brand from input features
- Top 3 brand predictions with confidence scores
- JSON API and web form interface
- Dockerized deployment with CI/CD
- MLflow experiment tracking for model performance
- Hyperparameter tuning via grid search

## Local Development

### Setup

```bash
pip install -r requirements.txt
```

### Train Model (Single Run)

Trains a Random Forest with default params (200 trees) and logs to MLflow:

```bash
python train.py
```

### Hyperparameter Tuning

Runs a grid search over 216 combinations (n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features) and logs each run:

```bash
python train.py --tune
```

### View MLflow Results

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open `http://localhost:5000` to compare runs by accuracy, precision, recall, and F1-score.

### Run Web App

```bash
python app.py
```

## API

**`POST /predict`**

```json
{
  "price": 30000,
  "mileage": 15000,
  "car_type": "SUV",
  "month": 6,
  "year": 2025
}
```

**Response**

```json
{
  "predicted_brand": "Hyundai",
  "top_brands": ["Hyundai: 41.50%", "Toyota: 24.00%", "Kia: 14.00%"]
}
```

## MLflow Tracking

Each training run logs:

| Artifact | Description |
|---|---|
| **Params** | n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features |
| **Metrics** | accuracy, precision_macro, recall_macro, f1_score_macro |
| **Model** | Serialized scikit-learn pipeline |

Results are stored in `mlflow.db` (SQLite backend).

## Deployment

Pushing to `main` triggers a GitHub Actions workflow that:

1. Copies files to EC2 via rsync
2. Builds a Docker image (model is trained during build)
3. Runs the container on port 80

### Secrets Required

| Secret | Description |
|---|---|
| `EC2_HOST` | EC2 public IP or DNS |
| `EC2_USERNAME` | SSH username (e.g. `ubuntu`) |
| `EC2_SSH_KEY` | Private SSH key content |
