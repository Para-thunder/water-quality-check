from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
import pandas as pd
import os

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
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
        print(f"Model loaded from {MODEL_PATH}")
    else:
        print(f"Warning: Model not found at {MODEL_PATH}. API will not work correctly.")

@app.post("/predict")
def predict(data: WaterData):
    if model is None:
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
