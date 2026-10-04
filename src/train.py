import os
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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

train_scenarios = sorted(
    df["scenario_id"].unique()
)[:96]

test_scenarios = sorted(
    df["scenario_id"].unique()
)[96:]

train_df = df[
    df["scenario_id"].isin(train_scenarios)
]

test_df = df[
    df["scenario_id"].isin(test_scenarios)
]

X_train = train_df[features]
y_train = train_df["vehicle_state"]

X_test = test_df[features]
y_test = test_df["vehicle_state"]

pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    ))
])

pipeline.fit(X_train, y_train)

os.makedirs("models", exist_ok=True)

joblib.dump(
    pipeline,
    "models/cats_model.pkl"
)

print("Training records:", len(train_df))
print("Testing records:", len(test_df))
print("Model saved to models/cats_model.pkl")