import joblib
import numpy as np

model = joblib.load("models/cats_model.pkl")

features = [
    "speed_kmh",
    "acceleration_ms2",
    "acceleration_change",
    "gyro_x",
    "gyro_y",
    "gyro_z",
    "impact_g",
    "battery_pct",
    "signal_strength_dbm",
    "motion_intensity",
    "absolute_acceleration_change",
    "speed_change"
]

def explain_prediction(data):

    motion_intensity = np.sqrt(
        data["gyro_x"] ** 2 +
        data["gyro_y"] ** 2 +
        data["gyro_z"] ** 2
    )

    absolute_acceleration_change = abs(
        data["acceleration_change"]
    )

    speed_change = data.get("speed_change", 0)

    record = {
        "speed_kmh": data["speed_kmh"],
        "acceleration_ms2": data["acceleration_ms2"],
        "acceleration_change": data["acceleration_change"],
        "gyro_x": data["gyro_x"],
        "gyro_y": data["gyro_y"],
        "gyro_z": data["gyro_z"],
        "impact_g": data["impact_g"],
        "battery_pct": data["battery_pct"],
        "signal_strength_dbm": data["signal_strength_dbm"],
        "motion_intensity": motion_intensity,
        "absolute_acceleration_change": absolute_acceleration_change,
        "speed_change": speed_change
    }

    X = np.array([
        [record[feature] for feature in features]
    ])

    prediction = model.predict(X)[0]

    probabilities = model.predict_proba(X)[0]

    classes = model.classes_

    confidence = {
        str(classes[i]): round(float(probabilities[i]), 4)
        for i in range(len(classes))
    }

    reasons = []

    if data["impact_g"] >= 0.5:
        reasons.append("High impact detected")

    if absolute_acceleration_change >= 3:
        reasons.append("Large acceleration change detected")

    if motion_intensity >= 1.5:
        reasons.append("High rotational motion detected")

    if data["speed_kmh"] >= 90:
        reasons.append("High vehicle speed detected")

    if data["signal_strength_dbm"] <= -100:
        reasons.append("Weak communication signal detected")

    if data["battery_pct"] <= 20:
        reasons.append("Low battery level detected")

    if not reasons:
        reasons.append("Telemetry values are within expected operating ranges")

    return {
        "prediction": prediction,
        "confidence": confidence,
        "reasons": reasons,
        "derived_features": {
            "motion_intensity": round(float(motion_intensity), 4),
            "absolute_acceleration_change": round(
                float(absolute_acceleration_change), 4
            ),
            "speed_change": round(float(speed_change), 4)
        }
    }


if __name__ == "__main__":

    sample = {
        "speed_kmh": 85,
        "acceleration_ms2": -2.1,
        "acceleration_change": 3.4,
        "gyro_x": 0.8,
        "gyro_y": 0.7,
        "gyro_z": 1.1,
        "impact_g": 0.3,
        "battery_pct": 72,
        "signal_strength_dbm": -70
    }

    result = explain_prediction(sample)

    print("Prediction:", result["prediction"])
    print("Confidence:", result["confidence"])
    print("Reasons:")

    for reason in result["reasons"]:
        print("-", reason)

    print("Derived Features:")
    print(result["derived_features"])