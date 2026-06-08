import json
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def test_index(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"Car Brand Predictor" in resp.data


def test_predict_html(client):
    resp = client.post("/predict", data={
        "price": "30000", "mileage": "25.5",
        "car_type": "SUV", "month": "6", "year": "2024",
    })
    assert resp.status_code == 200
    assert b"Predicted Brand" in resp.data or b"predicted_brand" in resp.data


def test_predict_json(client):
    resp = client.post("/predict", json={
        "price": 30000, "mileage": 25.5,
        "car_type": "SUV", "month": 6, "year": 2024,
    })
    assert resp.status_code == 200
    data = json.loads(resp.data)
    assert "predicted_brand" in data


def test_predict_invalid(client):
    resp = client.post("/predict", json={
        "price": "abc", "mileage": "xyz",
        "car_type": "SUV", "month": 1, "year": 2024,
    })
    assert resp.status_code == 400
