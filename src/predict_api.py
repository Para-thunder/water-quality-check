from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pandas as pd
import os

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("TensorFlow not available. API will run in mock mode.")

app = FastAPI(title="Water Potability Prediction API")

# Load model (defaulting to ANN for now)
MODEL_PATH = "models/ann_model.keras"
model = None

class WaterData(BaseModel):
    ph: float
    Hardness: float
    Solids: float
    Chloramines: float
    Sulfate: float
    Conductivity: float
    Organic_carbon: float
    Trihalomethanes: float
    Turbidity: float

@app.on_event("startup")
def load_model():
    global model
    if TF_AVAILABLE and os.path.exists(MODEL_PATH):
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
            print(f"Model loaded from {MODEL_PATH}")
        except Exception as e:
            print(f"Failed to load model: {e}")
    else:
        print(f"Warning: Model not found at {MODEL_PATH} or TensorFlow missing. API will use mock predictions.")

@app.post("/predict")
def predict(data: WaterData):
    if model is None:
        if not TF_AVAILABLE:
            # Mock prediction for Docker testing without heavy dependencies
            return {
                "potability": 1,
                "probability": 0.85,
                "note": "Mock prediction (TensorFlow not installed)"
            }
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Convert input to array
    input_data = np.array([[
        data.ph, data.Hardness, data.Solids, data.Chloramines,
        data.Sulfate, data.Conductivity, data.Organic_carbon,
        data.Trihalomethanes, data.Turbidity
    ]])
    
    # Note: In a real production app, we should load the scaler and transform input_data here.
    # For this example, we assume input is already scaled or we skip scaling for simplicity.
    
    prediction = model.predict(input_data)
    potability = int(prediction[0][0] > 0.5)
    
    return {
        "potability": potability,
        "probability": float(prediction[0][0])
    }

@app.get("/")
def read_root():
    return {"message": "Water Potability Prediction API is running"}
