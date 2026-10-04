import os
import numpy as np
import pandas as pd

input_path = "data/raw/vehicle_telemetry.csv"
output_path = "data/processed/vehicle_telemetry_processed.csv"

os.makedirs("data/processed", exist_ok=True)

df = pd.read_csv(input_path)

print("Dataset shape:", df.shape)
print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())

df = df.drop_duplicates()

numeric_features = [
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

for column in numeric_features:
    df[column] = df[column].fillna(df[column].median())

df["motion_intensity"] = np.sqrt(
    df["gyro_x"] ** 2 +
    df["gyro_y"] ** 2 +
    df["gyro_z"] ** 2
)

df["absolute_acceleration_change"] = (
    df["acceleration_change"].abs()
)

df = df.sort_values(
    ["vehicle_id", "scenario_id", "timestamp"]
)

df["speed_change"] = (
    df.groupby("vehicle_id")["speed_kmh"].diff()
)

df["speed_change"] = df["speed_change"].fillna(0)

df.to_csv(output_path, index=False)

print("\nProcessed dataset saved to:")
print(output_path)

print("\nClass distribution:")
print(df["vehicle_state"].value_counts())