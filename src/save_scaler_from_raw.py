"""Fit a StandardScaler on raw data and save it to models/scaler.pkl.

This replicates the cleaning/processing logic in `src/data_prep.py` so the scaler
maps raw inputs to the scaled features used for training.
"""
import os
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

RAW_PATH = os.path.join("data", "raw", "water_potability.csv")
SCALER_PATH = os.path.join("models", "scaler.pkl")


def load_and_clean(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    # Drop rows where target is missing
    if 'Potability' in df.columns:
        df = df.dropna(subset=['Potability'])
    # fill missing with mean
    df.fillna(df.mean(), inplace=True)
    df.fillna(0, inplace=True)
    return df


def fit_and_save_scaler(raw_path, scaler_path):
    print(f"Loading raw data from {raw_path}")
    df = load_and_clean(raw_path)
    if 'Potability' in df.columns:
        X = df.drop('Potability', axis=1)
    else:
        X = df
    scaler = StandardScaler()
    print("Fitting scaler on raw features...")
    scaler.fit(X)
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    joblib.dump(scaler, scaler_path)
    print(f"Saved scaler to {scaler_path}")


if __name__ == '__main__':
    fit_and_save_scaler(RAW_PATH, SCALER_PATH)
