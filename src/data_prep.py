import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import sys

def load_data(filepath):
    """Load data from a CSV file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    return pd.read_csv(filepath)

def clean_data(df):
    """Handle missing values by imputing with the mean."""
    df.fillna(df.mean(), inplace=True)
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
    
    os.makedirs(output_dir, exist_ok=True)
    
    train_path = os.path.join(output_dir, 'train.csv')
    test_path = os.path.join(output_dir, 'test.csv')
    
    print(f"Saving processed data to {output_dir}...")
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print("Data preparation complete.")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python src/data_prep.py <raw_data_path> <output_dir>")
        sys.exit(1)
        
    raw_path = sys.argv[1]
    out_dir = sys.argv[2]
    
    preprocess_data(raw_path, out_dir)
