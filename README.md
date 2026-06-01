# Car Brand Predictor

Flask web app that predicts car brand based on price, mileage, car type, month, and year using a Random Forest classifier.

## Features

- Predict car brand from input features
- Top 3 brand predictions with confidence scores
- JSON API and web form interface
- Dockerized deployment with CI/CD

## Local Development

```bash
pip install -r requirements.txt
python train.py
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
