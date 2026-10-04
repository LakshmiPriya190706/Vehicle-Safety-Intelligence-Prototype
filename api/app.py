import os
import sys

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from flask import Flask, request, jsonify
from src.explain import explain_prediction

app = Flask(__name__)

@app.route("/")
def home():
    return jsonify({
        "service": "CATS AI Vehicle State Classifier",
        "status": "running"
    })

@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    required_fields = [
        "speed_kmh",
        "acceleration_ms2",
        "acceleration_change",
        "gyro_x",
        "gyro_y",
        "gyro_z",
        "impact_g",
        "battery_pct",
        "signal_strength_dbm"
    ]

    missing_fields = [
        field for field in required_fields
        if field not in data
    ]

    if missing_fields:
        return jsonify({
            "error": "Missing required fields",
            "fields": missing_fields
        }), 400

    try:
        result = explain_prediction(data)

        return jsonify(result)

    except Exception as error:
        return jsonify({
            "error": str(error)
        }), 500


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )