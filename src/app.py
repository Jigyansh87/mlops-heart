from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
import logging
import time

# ------------------------
# Logging Configuration
# ------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# ------------------------
# App Initialization
# ------------------------
app = FastAPI(title="Heart Disease Prediction API")

START_TIME = time.time()
REQUEST_COUNT = 0

logger.info("Starting Heart Disease Prediction API")

# ------------------------
# Load Model Artifacts
# ------------------------
model = joblib.load("artifacts/random_forest_model.pkl")
scaler = joblib.load("artifacts/scaler.pkl")

logger.info("Model and scaler loaded successfully")

# ------------------------
# Request Schema
# ------------------------
class PatientInput(BaseModel):
    features: list

# ------------------------
# Health Endpoint
# ------------------------
@app.get("/health")
def health():
    logger.info("Health check endpoint called")
    return {"status": "ok"}

# ------------------------
# Metrics Endpoint (Simple Monitoring)
# ------------------------
@app.get("/metrics")
def metrics():
    uptime = round(time.time() - START_TIME, 2)
    return {
        "requests_served": REQUEST_COUNT,
        "uptime_seconds": uptime
    }

# ------------------------
# Prediction Endpoint
# ------------------------
@app.post("/predict")
def predict(data: PatientInput):
    global REQUEST_COUNT
    REQUEST_COUNT += 1

    logger.info(f"Received prediction request #{REQUEST_COUNT}")

    X = np.array(data.features).reshape(1, -1)
    X_scaled = scaler.transform(X)

    prob = model.predict_proba(X_scaled)[0][1]
    prediction = int(prob >= 0.5)

    logger.info(
        f"Prediction={prediction}, Confidence={round(float(prob), 3)}"
    )

    return {
        "prediction": "Disease" if prediction == 1 else "No Disease",
        "confidence": round(float(prob), 3)
    }
