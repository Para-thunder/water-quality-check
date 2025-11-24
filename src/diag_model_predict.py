import os
import joblib
import numpy as np
import pandas as pd

MODEL_PATH = os.path.join('models','ann_model.keras')
SCALER_PATH = os.path.join('models','scaler.pkl')

sample = [7.0,200.0,20000.0,7.0,300.0,400.0,15.0,60.0,4.0]

print('Model path exists:', os.path.exists(MODEL_PATH))
print('Scaler path exists:', os.path.exists(SCALER_PATH))

try:
    import tensorflow as tf
    model = tf.keras.models.load_model(MODEL_PATH)
    print('Loaded model, input_shape=', getattr(model,'input_shape',None))
except Exception as e:
    print('Failed loading model:', e)
    model = None

scaler = None
if os.path.exists(SCALER_PATH):
    try:
        scaler = joblib.load(SCALER_PATH)
        print('Loaded scaler, mean_:', getattr(scaler,'mean_',None))
    except Exception as e:
        print('Failed loading scaler:', e)

X = np.array([sample])
if scaler is not None:
    try:
        # if fitted with feature names, build dataframe
        if hasattr(scaler,'feature_names_in_'):
            X = pd.DataFrame(X, columns=scaler.feature_names_in_)
            Xs = scaler.transform(X)
        else:
            Xs = scaler.transform(X)
        print('Scaled sample:', Xs)
    except Exception as e:
        print('Scaler.transform failed:', e)
        Xs = X
else:
    Xs = X

if model is not None:
    try:
        # reshape if model expects 3d
        ish = getattr(model,'input_shape',None)
        xinp = Xs
        if ish and len(ish)==3 and xinp.ndim==2:
            xinp = xinp.reshape((xinp.shape[0], xinp.shape[1], 1))
        pred = model.predict(xinp)
        print('Raw model output:', pred)
        try:
            print('Probability float:', float(pred[0][0]))
        except Exception:
            print('Could not convert pred to float')
    except Exception as e:
        print('Model predict failed:', e)
else:
    print('No model to predict')
