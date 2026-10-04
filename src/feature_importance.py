import joblib
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(
    "data/processed/vehicle_telemetry_processed.csv"
)

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

model = joblib.load(
    "models/cats_model.pkl"
)

classifier = model.named_steps["model"]

importance = pd.Series(
    classifier.feature_importances_,
    index=features
).sort_values(ascending=True)

plt.figure(figsize=(8, 6))

importance.plot(
    kind="barh"
)

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("CATS AI Feature Importance")

plt.tight_layout()

plt.savefig(
    "results/feature_importance.png",
    dpi=200
)

plt.close()

print(importance.sort_values(ascending=False))