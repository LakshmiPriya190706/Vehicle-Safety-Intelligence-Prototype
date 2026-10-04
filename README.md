# CATS AI - Vehicle Telemetry Risk Classification System

CATS AI is a machine learning prototype that analyzes vehicle telemetry data and classifies the current vehicle state into three categories: NORMAL, RISK, and CRITICAL.

The project demonstrates a complete end-to-end machine learning workflow, starting from vehicle telemetry data and continuing through preprocessing, feature engineering, model training, evaluation, explainability, and REST API deployment.

The complete workflow is:

Vehicle Telemetry Dataset
        |
        v
Data Understanding and Quality Analysis
        |
        v
Data Preprocessing
        |
        v
Feature Engineering
        |
        v
Scenario-Based Train/Test Split
        |
        v
Random Forest Model Training
        |
        v
Model Evaluation
        |
        v
Feature Importance Analysis
        |
        v
Explainability Layer
        |
        v
Flask REST API
        |
        v
Vehicle State Prediction


## 1. Project Objective

The objective of CATS AI is to build a small machine learning prototype capable of analyzing vehicle telemetry and identifying the current vehicle state as:

- NORMAL
- RISK
- CRITICAL

The system uses multiple telemetry signals rather than depending on a single sensor.

The telemetry features include:

- Vehicle speed
- Acceleration
- Acceleration change
- Gyroscope measurements
- Impact level
- Battery percentage
- Communication signal strength

The system also provides an explanation for its prediction by identifying important telemetry conditions and displaying derived features used during inference.

This project is a technical prototype for demonstrating machine learning, data preprocessing, model evaluation, explainability, and backend integration. It is not intended to be used as a production accident detection or safety-critical system.


## 2. Dataset

The dataset is located at:

data/raw/vehicle_telemetry.csv

The dataset contains synthetic vehicle telemetry records.

The dataset contains the following major fields:

- timestamp
- vehicle_id
- scenario_id
- speed_kmh
- acceleration_ms2
- acceleration_change
- gyro_x
- gyro_y
- gyro_z
- impact_g
- battery_pct
- signal_strength_dbm
- vehicle_state

The target variable is:

vehicle_state

The possible target classes are:

- NORMAL
- RISK
- CRITICAL
## 3. Synthetic Dataset Disclosure

No real-world vehicle telemetry dataset was provided as part of the assessment.

Therefore, a synthetic dataset was used for this prototype.

The synthetic dataset was designed to simulate different vehicle operating scenarios and telemetry patterns associated with NORMAL, RISK, and CRITICAL states.

The dataset contains multiple vehicles and multiple scenarios so that the model can be evaluated on unseen scenarios.

The synthetic nature of the dataset is explicitly disclosed because the resulting model performance should not be interpreted as real-world accident-detection performance.

For a production implementation, the model would need to be trained and validated using appropriately collected and labeled real-world vehicle telemetry.

## 4. Technologies Used

The project uses:

- Python
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Joblib
- Flask
- Visual Studio Code

The machine learning model used is:

Random Forest Classifier

The backend API is implemented using:

Flask

## 5. Environment Setup

The project was developed and tested using a Python virtual environment.

Python should be installed before starting the project.

To verify Python:

python --version

A Python version that is compatible with the installed packages should be used.

Open PowerShell and navigate to the project directory:

cd C:\CATS_AI

Create a virtual environment:

python -m venv .venv

Activate the virtual environment:

.\.venv\Scripts\Activate.ps1

After successful activation, the terminal should display:

(.venv) PS C:\CATS_AI>

## 6. Installing Dependencies

The project includes a requirements.txt file containing the required Python packages.

Install all dependencies using:

pip install -r requirements.txt

The required packages are:

numpy
pandas
scikit-learn
matplotlib
joblib
flask

The environment can be verified using:

python -c "import numpy, pandas, sklearn, matplotlib, joblib, flask; print('ALL REQUIRED PACKAGES OK')"

Expected output:

ALL REQUIRED PACKAGES OK

## 7. Running the Project from the Beginning

The project should be executed from the root directory:

C:\CATS_AI

The complete execution sequence is:

1. Activate the virtual environment.
2. Preprocess the dataset.
3. Train the machine learning model.
4. Evaluate the model.
5. Generate feature importance.
6. Test the explainability layer.
7. Start the Flask API.
8. Send telemetry to the prediction endpoint.
9. Inspect the prediction and explanation.
## 8. Step 1 - Activate the Environment

Open PowerShell in the project directory:

cd C:\CATS_AI

Activate the virtual environment:

.\.venv\Scripts\Activate.ps1

The terminal should show:

(.venv) PS C:\CATS_AI>
## 9. Step 2 - Preprocess the Dataset

The preprocessing code is located at:

src/preprocess.py

Run:

python src\preprocess.py

The preprocessing stage loads:

data/raw/vehicle_telemetry.csv

The script performs data-quality checks including:

- Dataset shape inspection
- Missing-value inspection
- Duplicate-row inspection
- Numerical feature preparation

The script also performs feature engineering.

The generated features include:

- motion_intensity
- absolute_acceleration_change
- speed_change

The processed dataset is saved to:

data/processed/vehicle_telemetry_processed.csv

The processed dataset is then used by the model-training stage.

## 11. Feature Engineering

The model does not rely only on the original telemetry measurements.

Additional features are derived to represent vehicle motion more effectively.

### Motion Intensity

The three gyroscope measurements are combined into a single rotational-motion feature:

motion_intensity = sqrt(gyro_x² + gyro_y² + gyro_z²)

This represents the overall magnitude of rotational motion.

### Absolute Acceleration Change

The magnitude of acceleration change is calculated using:

absolute_acceleration_change = abs(acceleration_change)

This captures sudden changes in acceleration regardless of whether the change is positive or negative.

### Speed Change

Speed change is calculated from sequential vehicle telemetry observations.

The calculation is performed within the vehicle scenario so that the beginning of a new scenario does not incorrectly inherit the previous scenario's speed.
## 10. Step 3 - Train the Machine Learning Model

The model-training code is located at:

src/train.py

Run:

python src\train.py

The training process loads:

data/processed/vehicle_telemetry_processed.csv

The features used by the model are:

- speed_kmh
- acceleration_ms2
- acceleration_change
- gyro_x
- gyro_y
- gyro_z
- impact_g
- battery_pct
- signal_strength_dbm
- motion_intensity
- absolute_acceleration_change
- speed_change

The target variable is:

vehicle_state

The model used is a Random Forest Classifier.

The main configuration is:

n_estimators = 200
random_state = 42
class_weight = balanced

The Random Forest model was selected because it can capture nonlinear relationships between telemetry signals, works well with numerical features, provides feature importance, and provides class probability estimates through predict_proba().

The trained model is saved as:

models/cats_model.pkl

## 11. Step 4 - Evaluate the Model

The evaluation code is located at:

src/evaluate.py

Run:

python src\evaluate.py

The evaluation generates:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

The classification report is saved to:

results/classification_report.txt

The confusion matrix is saved to:

results/confusion_matrix.png

The evaluation should not rely only on accuracy.

For this application, recall is especially important because incorrectly classifying a genuinely CRITICAL event as NORMAL can have more serious consequences than generating an additional warning for a NORMAL event.

Therefore, the CRITICAL class recall and the confusion matrix should be examined carefully.


## 12. Step 5 - Generate Feature Importance

The feature-importance code is located at:

src/feature_importance.py

Run:

python src\feature_importance.py

This extracts feature importance values from the trained Random Forest model.

The visualization is saved to:

results/feature_importance.png

Feature importance provides an additional level of model interpretability by showing which telemetry features contributed most strongly to the model's decisions.

## 13. Step 6 - Test the Explainability Layer

The explainability code is located at:

src/explain.py

Run:

python src\explain.py

The explainability layer:

1. Loads the trained model.
2. Accepts a telemetry record.
3. Calculates derived features.
4. Generates a vehicle-state prediction.
5. Calculates class probabilities.
6. Checks important telemetry conditions.
7. Generates human-readable reasons.

The output contains:

- Prediction
- Confidence
- Reasons
- Derived features

An example result is:

Prediction: RISK

Confidence:
CRITICAL = 0.115
NORMAL = 0.000
RISK = 0.885

Derived Features:
absolute_acceleration_change = 3.4
motion_intensity = 1.5297
speed_change = 0.0

Reasons:
Large acceleration change detected
## 14. Understanding the Confidence Values

The model uses Random Forest class probabilities through:

predict_proba()

For example:

CRITICAL = 0.115
NORMAL = 0.000
RISK = 0.885

This indicates that the model strongly favors the RISK class for that particular telemetry input.

These values represent model-estimated class probabilities.

They should not be interpreted as a literal probability that an accident will occur.

A production safety system would require probability calibration and validation against real-world labeled events before such probabilities could be used for operational decisions.
## 15. Prediction Module

The reusable prediction logic is contained in:

src/predict.py

This module loads:

models/cats_model.pkl

and provides a reusable prediction function.

The prediction process is:

Input Telemetry
       |
       v
Derived Features
       |
       v
Random Forest Model
       |
       v
Predicted Vehicle State
       |
       v
Class Probabilities


## 16. Step 7 - Flask API

The Flask backend is implemented in:

api/app.py

The API provides two main endpoints:

GET /
POST /predict

The Flask API connects the trained machine learning model to an external client.

This allows a telemetry record to be submitted through an HTTP request and receive a machine learning prediction as a JSON response.

## 17. Starting the Flask API

From the project root:

C:\CATS_AI

run:

python api\app.py

The Flask server should start at:

http://127.0.0.1:5000

The terminal should display something similar to:

* Running on http://127.0.0.1:5000

Keep this terminal running while testing the API.

Open a second terminal for sending prediction requests.
## 18. Testing the Flask Health Endpoint

Open the following address in a browser:

http://127.0.0.1:5000/

The API should return:

{
    "service": "CATS AI Vehicle State Classifier",
    "status": "running"
}

This confirms that:

- Flask is running.
- app.py was loaded successfully.
- The project imports are working.
- The API server is listening on port 5000.
## 19. Testing the Prediction API

The main prediction endpoint is:

POST /predict

The endpoint accepts vehicle telemetry in JSON format.

Example telemetry:

{
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
## 20. Testing /predict Using PowerShell

Keep the Flask server running in the first terminal.

Open a second VS Code terminal.

Activate the virtual environment if necessary:

.\.venv\Scripts\Activate.ps1

Create the request body:

$body = @{
    speed_kmh = 85
    acceleration_ms2 = -2.1
    acceleration_change = 3.4
    gyro_x = 0.8
    gyro_y = 0.7
    gyro_z = 1.1
    impact_g = 0.3
    battery_pct = 72
    signal_strength_dbm = -70
} | ConvertTo-Json

Send the request:

Invoke-RestMethod -Uri "http://127.0.0.1:5000/predict" -Method POST -ContentType "application/json" -Body $body

The API returns the prediction and explanation.
## 21. Example API Response

An example response from the working prototype is:

prediction:
RISK

confidence:
CRITICAL = 0.115
NORMAL = 0.000
RISK = 0.885

derived_features:
absolute_acceleration_change = 3.4
motion_intensity = 1.5297
speed_change = 0.0

reasons:
Large acceleration change detected

This means that the Random Forest model classified the supplied telemetry as RISK and assigned the highest estimated class probability to RISK.
## 22. Testing Different Vehicle Conditions

The API can be tested with different telemetry values.

For example, relatively stable telemetry can be used to represent a normal operating condition:

{
    "speed_kmh": 40,
    "acceleration_ms2": 0.2,
    "acceleration_change": 0.1,
    "gyro_x": 0.05,
    "gyro_y": 0.05,
    "gyro_z": 0.08,
    "impact_g": 0.02,
    "battery_pct": 85,
    "signal_strength_dbm": -60
}

A higher-risk telemetry condition can contain stronger acceleration changes and rotational motion:

{
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

A severe telemetry condition can contain very high speed, acceleration changes, rotational motion, and impact:

{
    "speed_kmh": 110,
    "acceleration_ms2": -5.5,
    "acceleration_change": 6.5,
    "gyro_x": 2.5,
    "gyro_y": 2.2,
    "gyro_z": 2.8,
    "impact_g": 1.2,
    "battery_pct": 65,
    "signal_strength_dbm": -80
}

The model determines the final class based on the patterns learned during training. The prediction should not be manually forced to a particular class.
## 23. Using the Project by Cloning a Git Repository

If the project is uploaded to a Git repository, it can be cloned using:

git clone <REPOSITORY_URL>

For example:

git clone https://github.com/your-username/CATS_AI.git

Enter the project directory:

cd CATS_AI

Create the virtual environment:

python -m venv .venv

Activate it:

.\.venv\Scripts\Activate.ps1

Install dependencies:

pip install -r requirements.txt

Run the preprocessing:

python src\preprocess.py

Train the model:

python src\train.py

Evaluate:

python src\evaluate.py

Generate feature importance:

python src\feature_importance.py

Test explainability:

python src\explain.py

Start the Flask API:

python api\app.py

The repository URL should be replaced with the actual repository URL when the project is uploaded to GitHub or another Git hosting platform.

## 24. False Alarm Analysis

A false alarm occurs when the system predicts an elevated-risk state while the actual vehicle condition is normal.

Possible causes include:

- Temporary sensor spikes
- Sudden but legitimate acceleration
- Strong rotational movement during normal driving
- Noisy impact measurements
- Unusual but safe driving conditions
- Differences between synthetic data and real-world telemetry

Possible improvements include:

- Sensor filtering
- Temporal smoothing
- Rolling-window features
- More realistic negative examples
- Real-world labeled telemetry
- Probability calibration
- Multiple consecutive prediction confirmation
- Context-aware thresholds
## 25. Why Random Forest Was Selected

Random Forest was selected for this prototype because:

- It can model nonlinear relationships.
- It works well with numerical telemetry features.
- It is relatively robust to noisy features.
- It does not require feature scaling.
- It supports class weighting.
- It provides feature importance.
- It provides class probability estimates.
- It provides a strong baseline for a relatively small prototype dataset.

A more advanced model could be evaluated if larger real-world sequential datasets become available.

## 26. Limitations

This is a prototype and has several limitations.

### Synthetic Dataset

The dataset is synthetic and cannot represent every real-world driving condition.

### Limited Sensor Information

The current system uses a limited number of telemetry features.

Additional vehicle signals could improve the model.

### No Temporal Deep Learning

The current model primarily processes individual telemetry observations.

A production implementation could use sequential models such as:

- LSTM
- GRU
- Temporal Convolutional Networks
- Transformer-based time-series models

### No Physical Vehicle Integration

The Flask API simulates telemetry ingestion but is not connected directly to physical vehicle sensors.

### No Production Deployment

The Flask server is intended for local development and prototype demonstration.

A production deployment would require a production WSGI server, authentication, logging, monitoring, security controls, and infrastructure appropriate for the application.



## 27. Generated Results

The project generates the following important outputs:

data/processed/vehicle_telemetry_processed.csv

models/cats_model.pkl

results/classification_report.txt

results/confusion_matrix.png

results/feature_importance.png

These files provide the processed data, trained model, evaluation metrics, confusion matrix, and feature-importance visualization.