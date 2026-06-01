import os
from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "car_brand_model.joblib")
MODEL = None
FEATURES = ["price", "mileage", "car_type", "month", "year"]


def load_model():
    global MODEL
    if MODEL is None:
        MODEL = joblib.load(MODEL_PATH)


@app.route("/", methods=["GET"])
def index():
    car_types = ["Sedan", "SUV", "Hatchback", "Coupe", "Pickup", "Convertible"]
    months = list(range(1, 13))
    years = list(range(2016, 2026))
    return render_template("index.html", car_types=car_types, months=months, years=years)


@app.route("/predict", methods=["POST"])
def predict():
    if request.is_json:
        payload = request.get_json()
    else:
        payload = request.form.to_dict()

    try:
        price = float(payload.get("price", ""))
        mileage = float(payload.get("mileage", ""))
        car_type = payload.get("car_type", "SUV")
        month = int(payload.get("month", 1))
        year = int(payload.get("year", 2025))
    except (ValueError, TypeError):
        return jsonify({"error": "Price and mileage must be numeric."}), 400

    df = pd.DataFrame([
        {
            "price": price,
            "mileage": mileage,
            "car_type": car_type,
            "month": month,
            "year": year,
        }
    ])

    if MODEL is None:
        load_model()

    if MODEL is None:
        return jsonify({"error": "Model not loaded."}), 500

    prediction = MODEL.predict(df)[0]
    if hasattr(MODEL, "predict_proba"):
        scores = MODEL.predict_proba(df)[0]
        top_scores = sorted(
            zip(MODEL.classes_, scores), key=lambda x: x[1], reverse=True
        )[:3]
        top_brands = [f"{brand}: {score:.2%}" for brand, score in top_scores]
    else:
        top_brands = []

    result = {
        "predicted_brand": prediction,
        "top_brands": top_brands,
        "input": {
            "price": price,
            "mileage": mileage,
            "car_type": car_type,
            "month": month,
            "year": year,
        },
    }

    if request.is_json:
        return jsonify(result)

    return render_template(
        "index.html",
        result=result,
        car_types=["Sedan", "SUV", "Hatchback", "Coupe", "Pickup", "Convertible"],
        months=list(range(1, 13)),
        years=list(range(2016, 2026)),
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
