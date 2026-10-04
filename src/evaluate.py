import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

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

test_scenarios = sorted(
    df["scenario_id"].unique()
)[96:]

test_df = df[
    df["scenario_id"].isin(test_scenarios)
]

X_test = test_df[features]
y_test = test_df["vehicle_state"]

model = joblib.load(
    "models/cats_model.pkl"
)

predictions = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)

report = classification_report(
    y_test,
    predictions
)

print("Accuracy:", accuracy)
print("\nClassification Report:")
print(report)

os.makedirs("results", exist_ok=True)

with open(
    "results/classification_report.txt",
    "w"
) as file:
    file.write(f"Accuracy: {accuracy}\n\n")
    file.write(report)

labels = [
    "NORMAL",
    "RISK",
    "CRITICAL"
]

matrix = confusion_matrix(
    y_test,
    predictions,
    labels=labels
)

display = ConfusionMatrixDisplay(
    confusion_matrix=matrix,
    display_labels=labels
)

display.plot()

plt.title("CATS AI Confusion Matrix")
plt.tight_layout()

plt.savefig(
    "results/confusion_matrix.png",
    dpi=200
)

plt.close()

print("Evaluation results saved to results/")