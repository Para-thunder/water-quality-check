from flask import Flask, request, jsonify, render_template
import numpy as np
import os
import joblib
import pandas as pd

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("TensorFlow not available. API will run in mock mode.")

app = Flask(__name__)

# Load model (defaulting to ANN for now)
MODEL_PATH = "models/ann_model.keras"
model = None
scaler = None
SCALER_PATH = "models/scaler.pkl"

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

# Load model on startup
load_model()
# Load scaler if available
if os.path.exists(SCALER_PATH):
    try:
        scaler = joblib.load(SCALER_PATH)
        print(f"Scaler loaded from {SCALER_PATH}")
    except Exception as e:
        print(f"Warning: Failed to load scaler: {e}")
else:
    print("No scaler found at models/scaler.pkl; inputs will be used raw.")

# Print model and scaler diagnostics to help debugging
if model is not None:
    try:
        input_shape = getattr(model, 'input_shape', None)
        print(f"Model input_shape: {input_shape}")
        try:
            model.summary()
        except Exception:
            print("(Could not print model.summary())")
    except Exception:
        pass

if scaler is not None:
    try:
        mean = getattr(scaler, 'mean_', None)
        scale = getattr(scaler, 'scale_', None)
        var = getattr(scaler, 'var_', None)
        print(f"Scaler mean: {mean}")
        print(f"Scaler scale: {scale}")
    except Exception:
        print("(Could not read scaler parameters)")


@app.route("/info", methods=["GET"])
def info():
    """Return model/scaler basic info for debugging."""
    info = {}
    if model is not None:
        info['model_input_shape'] = getattr(model, 'input_shape', None)
        try:
            # serialize basic layer info
            info['model_layers'] = [type(l).__name__ for l in model.layers]
        except Exception:
            info['model_layers'] = None
    else:
        info['model'] = None

    if scaler is not None:
        info['scaler_mean'] = getattr(scaler, 'mean_', None).tolist() if hasattr(scaler, 'mean_') else None
        info['scaler_scale'] = getattr(scaler, 'scale_', None).tolist() if hasattr(scaler, 'scale_') else None
    else:
        info['scaler'] = None

    return jsonify(info)

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict_ui", methods=["POST"])
def predict_ui():
    global model
    try:
        # Extract features from form
        features = [
            float(request.form.get("ph", 0)),
            float(request.form.get("Hardness", 0)),
            float(request.form.get("Solids", 0)),
            float(request.form.get("Chloramines", 0)),
            float(request.form.get("Sulfate", 0)),
            float(request.form.get("Conductivity", 0)),
            float(request.form.get("Organic_carbon", 0)),
            float(request.form.get("Trihalomethanes", 0)),
            float(request.form.get("Turbidity", 0))
        ]
        
        # Debug: Print input features
        print(f"[UI] Input features: {features}")
        if model is None:
            if not TF_AVAILABLE:
                potability = 1
                probability = 0.85
            else:
                return render_template("index.html", prediction_text="Error: Model not loaded", result_class="not-potable")
        else:
            input_data = np.array([features])
            # Apply scaler if available (use DataFrame with feature names if scaler was fitted with them)
            if scaler is not None:
                try:
                    if hasattr(scaler, 'feature_names_in_'):
                        input_df = pd.DataFrame(input_data, columns=scaler.feature_names_in_)
                        input_data = scaler.transform(input_df)
                    else:
                        input_data = scaler.transform(input_data)
                    print("[UI] Applied scaler to input data")
                except Exception as e:
                    print(f"[UI] Warning: scaler.transform failed: {e}")
            # If model expects 3D input (e.g., CNN), reshape accordingly
            try:
                input_shape = getattr(model, 'input_shape', None)
                if input_shape and len(input_shape) == 3 and input_data.ndim == 2:
                    input_data = input_data.reshape((input_data.shape[0], input_data.shape[1], 1))
            except Exception:
                pass
            prediction = model.predict(input_data)
            probability = float(prediction[0][0])
            potability = int(probability > 0.5)
            # Debug: Print prediction probability
            print(f"[UI] Prediction probability: {probability}")
            if probability == 0.0:
                print("[UI] Warning: Model predicted 0.00 probability. Check model and input scaling.")

        result_text = "Water is Potable!" if potability == 1 else "Water is Not Potable."
        result_class = "potable" if potability == 1 else "not-potable"
        return render_template("index.html", 
                             prediction_text=result_text, 
                             confidence=f"{probability:.2f}",
                             result_class=result_class)

    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}", result_class="not-potable")

@app.route("/predict", methods=["POST"])
def predict():
    global model
    
    # Get JSON data
    data = request.get_json()
    
    if not data:
        return jsonify({"error": "No input data provided"}), 400

    # Check if model is loaded
    if model is None:
        if not TF_AVAILABLE:
            # Mock prediction for Docker testing without heavy dependencies
            return jsonify({
                "potability": 1,
                "probability": 0.85,
                "note": "Mock prediction (TensorFlow not installed)"
            })
        return jsonify({"error": "Model not loaded"}), 503
    
    try:
        # Extract features in correct order
        features = [
            float(data.get("ph", 0)),
            float(data.get("Hardness", 0)),
            float(data.get("Solids", 0)),
            float(data.get("Chloramines", 0)),
            float(data.get("Sulfate", 0)),
            float(data.get("Conductivity", 0)),
            float(data.get("Organic_carbon", 0)),
            float(data.get("Trihalomethanes", 0)),
            float(data.get("Turbidity", 0))
        ]
        # Debug: Print input features
        print(f"[API] Input features: {features}")
        # Convert input to array
        input_data = np.array([features])
        # Apply scaler if available (preserve feature names to avoid sklearn warning)
        if scaler is not None:
            try:
                if hasattr(scaler, 'feature_names_in_'):
                    input_df = pd.DataFrame(input_data, columns=scaler.feature_names_in_)
                    input_data = scaler.transform(input_df)
                else:
                    input_data = scaler.transform(input_data)
                print("[API] Applied scaler to input data")
            except Exception as e:
                print(f"[API] Warning: scaler.transform failed: {e}")
        # Reshape for CNN-style models if needed
        try:
            input_shape = getattr(model, 'input_shape', None)
            if input_shape and len(input_shape) == 3 and input_data.ndim == 2:
                input_data = input_data.reshape((input_data.shape[0], input_data.shape[1], 1))
        except Exception:
            pass
        # Predict
        prediction = model.predict(input_data)
        probability = float(prediction[0][0])
        potability = int(probability > 0.5)
        # Debug: Print prediction probability
        print(f"[API] Prediction probability: {probability}")
        if probability == 0.0:
            print("[API] Warning: Model predicted 0.00 probability. Check model and input scaling.")
        return jsonify({
            "potability": potability,
            "probability": probability
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
