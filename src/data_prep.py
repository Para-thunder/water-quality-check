import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os
import sys

def load_data(filepath):
    """Load data from a CSV file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    return pd.read_csv(filepath)

def clean_data(df):
    """Handle missing values."""
    # Drop rows where target is missing
    df = df.dropna(subset=['Potability'])
    
    # Fill missing feature values with the mean
    df.fillna(df.mean(), inplace=True)
    
    # If any NaNs remain (e.g., if a column was all NaNs), fill with 0
    df.fillna(0, inplace=True)
    
    return df

def preprocess_data(raw_data_path, output_dir):
    """Load, clean, scale, and split data."""
    print("Loading data...")
    df = load_data(raw_data_path)
    
    print("Cleaning data...")
    df = clean_data(df)
    
    X = df.drop('Potability', axis=1)
    y = df['Potability']
    
    print("Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Combine X and y for saving
    train_df = pd.concat([X_train, y_train.reset_index(drop=True)], axis=1)
    test_df = pd.concat([X_test, y_test.reset_index(drop=True)], axis=1)
    
    # Ensure no NaNs in the output
    train_df.fillna(0, inplace=True)
    test_df.fillna(0, inplace=True)
    
    os.makedirs(output_dir, exist_ok=True)
    
    train_path = os.path.join(output_dir, 'train.csv')
    test_path = os.path.join(output_dir, 'test.csv')
    
    print(f"Saving processed data to {output_dir}...")
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    # Save the fitted scaler so we can apply the same transformation at inference time
    os.makedirs("models", exist_ok=True)
    scaler_path = os.path.join("models", "scaler.pkl")
    try:
        joblib.dump(scaler, scaler_path)
        print(f"Saved scaler to {scaler_path}")
    except Exception as e:
        print(f"Warning: Failed to save scaler: {e}")
    
    print("Data preparation complete.")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python src/data_prep.py <raw_data_path> <output_dir>")
        sys.exit(1)
        
    raw_path = sys.argv[1]
    out_dir = sys.argv[2]
    
    preprocess_data(raw_path, out_dir)
