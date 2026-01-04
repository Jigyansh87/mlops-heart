# src/app.py

import os
import sys
import joblib
import numpy as np
from typing import List

from fastapi import FastAPI
from pydantic import BaseModel

# -------------------------------------------------------------------
# Absolute paths inside Docker / Kubernetes container
# -------------------------------------------------------------------
MODEL_PATH = "/app/artifacts/random_forest_model.pkl"
SCALER_PATH = "/app/artifacts/scaler.pkl"

# -------------------------------------------------------------------
# Validate artifacts at startup (prevents CrashLoopBackOff)
# -------------------------------------------------------------------
if not os.path.exists(MODEL_PATH):
    print(f"ERROR: Model file not found at {MODEL_PATH}")
    sys.exit(1)

if not os.path.exists(SCALER_PATH):
    print(f"ERROR: Scaler file not found at {SCALER_PATH}")
    sys.exit(1)

# -------------------------------------------------------------------
# Load model and scaler
# -------------------------------------------------------------------
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# -------------------------------------------------------------------
# FastAPI app
# -------------------------------------------------------------------
app = FastAPI(title="Heart Disease Prediction API")

# -------------------------------------------------------------------
# Request schema
# -------------------------------------------------------------------
class PatientInput(BaseModel):
    features: List[float]

# -------------------------------------------------------------------
# Health check endpoint
# -------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}

# -------------------------------------------------------------------
# Prediction endpoint
# -------------------------------------------------------------------
@app.post("/predict")
def predict(data: PatientInput):
    X = np.array(data.features).reshape(1, -1)
    X_scaled = scaler.transform(X)

    prob = model.predict_proba(X_scaled)[0][1]
    prediction = int(prob >= 0.5)

    return {
        "prediction": "Disease" if prediction == 1 else "No Disease",
        "confidence": round(float(prob), 3)
    }
