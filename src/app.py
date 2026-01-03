# src/app.py
import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

# Load model and scaler
model = joblib.load("artifacts/random_forest_model.pkl")
scaler = joblib.load("artifacts/scaler.pkl")

app = FastAPI(title="Heart Disease Prediction API")

# Input schema
class PatientInput(BaseModel):
    features: list

# Health check
@app.get("/health")
def health():
    return {"status": "ok"}

# Prediction endpoint
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
