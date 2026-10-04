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

def predict_vehicle_state(data):

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

    return prediction, confidence