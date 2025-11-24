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
ANN_MODEL_PATH = "models/ann_model.keras"
RF_MODEL_PATH = "models/rf_model.pkl"
ann_model = None
rf_model = None
scaler = None
SCALER_PATH = "models/scaler.pkl"

def load_model():
    global ann_model, rf_model
    # Load ANN
    if TF_AVAILABLE and os.path.exists(ANN_MODEL_PATH):
        try:
            ann_model = tf.keras.models.load_model(ANN_MODEL_PATH)
            print(f"ANN Model loaded from {ANN_MODEL_PATH}")
        except Exception as e:
            print(f"Failed to load ANN model: {e}")
    else:
        print(f"Warning: ANN Model not found at {ANN_MODEL_PATH} or TensorFlow missing.")

    # Load RF
    if os.path.exists(RF_MODEL_PATH):
        try:
            rf_model = joblib.load(RF_MODEL_PATH)
            print(f"RF Model loaded from {RF_MODEL_PATH}")
        except Exception as e:
            print(f"Failed to load RF model: {e}")
    else:
        print(f"Warning: RF Model not found at {RF_MODEL_PATH}")

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
if ann_model is not None:
    try:
        input_shape = getattr(ann_model, 'input_shape', None)
        print(f"ANN Model input_shape: {input_shape}")
        try:
            ann_model.summary()
        except Exception:
            print("(Could not print ANN model.summary())")
    except Exception:
        pass

if rf_model is not None:
    print("RF Model loaded successfully")

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
    if ann_model is not None:
        info['ann_model_input_shape'] = getattr(ann_model, 'input_shape', None)
        try:
            # serialize basic layer info
            info['ann_model_layers'] = [type(l).__name__ for l in ann_model.layers]
        except Exception:
            info['ann_model_layers'] = None
    else:
        info['ann_model'] = None

    if rf_model is not None:
        info['rf_model'] = "Loaded"
    else:
        info['rf_model'] = None

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
        # Use a dictionary to map form fields to feature names explicitly
        form_data = request.form.to_dict()
        
        # Define the expected features (names must match what the scaler expects)
        # We'll try to get this from the scaler if possible, otherwise default to the known list
        expected_features = [
            "ph", "Hardness", "Solids", "Chloramines", "Sulfate", 
            "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity"
        ]
        
        if scaler is not None and hasattr(scaler, 'feature_names_in_'):
            expected_features = list(scaler.feature_names_in_)
            
        # Construct input dictionary, handling missing values
        input_dict = {}
        for feature in expected_features:
            val = form_data.get(feature)
            if val and val.strip():
                input_dict[feature] = float(val)
            else:
                # If missing, use scaler mean if available, else 0
                if scaler is not None and hasattr(scaler, 'mean_') and hasattr(scaler, 'feature_names_in_'):
                    try:
                        idx = list(scaler.feature_names_in_).index(feature)
                        mean_val = scaler.mean_[idx]
                        input_dict[feature] = mean_val
                        print(f"[UI] Warning: Missing value for {feature}, using mean: {mean_val}")
                    except ValueError:
                        input_dict[feature] = 0.0
                else:
                    input_dict[feature] = 0.0
                    print(f"[UI] Warning: Missing value for {feature}, defaulting to 0.0")

        # Create DataFrame to ensure correct column order for scaler
        input_df = pd.DataFrame([input_dict])
        
        # Debug: Print input features
        print(f"[UI] Input features: {input_dict}")
        
        if ann_model is None and rf_model is None:
            if not TF_AVAILABLE:
                # Mock
                ann_potability = 1
                ann_probability = 0.85
                rf_potability = 1
                rf_probability = 0.82
            else:
                return render_template("index.html", prediction_text="Error: No models loaded", result_class="not-potable")
        else:
            # Apply scaler
            if scaler is not None:
                try:
                    input_data = scaler.transform(input_df)
                    print("[UI] Applied scaler to input data")
                except Exception as e:
                    print(f"[UI] Warning: scaler.transform failed: {e}")
                    input_data = input_df.values
            else:
                input_data = input_df.values

            # ANN Prediction
            ann_potability = None
            ann_probability = None
            if ann_model is not None:
                try:
                    # If model expects 3D input (e.g., CNN), reshape accordingly
                    input_shape = getattr(ann_model, 'input_shape', None)
                    ann_input = input_data.copy()
                    if input_shape and len(input_shape) == 3 and ann_input.ndim == 2:
                        ann_input = ann_input.reshape((ann_input.shape[0], ann_input.shape[1], 1))
                    
                    prediction = ann_model.predict(ann_input)
                    ann_probability = float(prediction[0][0])
                    ann_potability = int(ann_probability > 0.5)
                    print(f"[UI] ANN Prediction probability: {ann_probability}")
                except Exception as e:
                    print(f"[UI] ANN Prediction failed: {e}")

            # RF Prediction
            rf_potability = None
            rf_probability = None
            if rf_model is not None:
                try:
                    # RF expects 2D input
                    rf_input = input_data
                    rf_pred = rf_model.predict(rf_input)
                    rf_prob = rf_model.predict_proba(rf_input)[:, 1]
                    rf_probability = float(rf_prob[0])
                    rf_potability = int(rf_pred[0])
                    print(f"[UI] RF Prediction probability: {rf_probability}")
                except Exception as e:
                    print(f"[UI] RF Prediction failed: {e}")

        return render_template("index.html", 
                             ann_prediction="Potable" if ann_potability == 1 else "Not Potable",
                             ann_confidence=f"{ann_probability * 100:.1f}%" if ann_probability is not None else "N/A",
                             ann_class="potable" if ann_potability == 1 else "not-potable",
                             rf_prediction="Potable" if rf_potability == 1 else "Not Potable",
                             rf_confidence=f"{rf_probability * 100:.1f}%" if rf_probability is not None else "N/A",
                             rf_class="potable" if rf_potability == 1 else "not-potable",
                             form_data=form_data)

    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}", result_class="not-potable", form_data=request.form)

@app.route("/predict", methods=["POST"])
def predict():
    global ann_model
    
    # Get JSON data
    data = request.get_json()
    
    if not data:
        return jsonify({"error": "No input data provided"}), 400

    # Check if model is loaded
    if ann_model is None:
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
            input_shape = getattr(ann_model, 'input_shape', None)
            if input_shape and len(input_shape) == 3 and input_data.ndim == 2:
                input_data = input_data.reshape((input_data.shape[0], input_data.shape[1], 1))
        except Exception:
            pass
        # Predict
        prediction = ann_model.predict(input_data)
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
