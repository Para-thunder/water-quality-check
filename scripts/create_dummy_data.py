import pandas as pd
import numpy as np
import os

def create_dummy_data(output_path):
    """Create a dummy water potability dataset for testing."""
    np.random.seed(42)
    n_samples = 100
    
    data = {
        'ph': np.random.uniform(0, 14, n_samples),
        'Hardness': np.random.uniform(100, 300, n_samples),
        'Solids': np.random.uniform(5000, 50000, n_samples),
        'Chloramines': np.random.uniform(0, 15, n_samples),
        'Sulfate': np.random.uniform(200, 500, n_samples),
        'Conductivity': np.random.uniform(200, 700, n_samples),
        'Organic_carbon': np.random.uniform(0, 30, n_samples),
        'Trihalomethanes': np.random.uniform(0, 120, n_samples),
        'Turbidity': np.random.uniform(1, 7, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Create a pattern for the model to learn:
    # Water is potable (1) if pH is neutral (6.5-8.5) AND Sulfate is low (< 400)
    condition = (df['ph'] > 6.5) & (df['ph'] < 8.5) & (df['Sulfate'] < 400)
    df['Potability'] = np.where(condition, 1, 0)
    
    # Add 10% noise (flip labels) to make it realistic
    mask = np.random.choice([True, False], size=n_samples, p=[0.1, 0.9])
    df['Potability'] = np.where(mask, 1 - df['Potability'], df['Potability'])
    
    # Introduce some missing values
    mask_nan = np.random.choice([True, False], size=df.shape, p=[0.1, 0.9])
    df = df.mask(mask_nan)
    # Ensure target has no missing values
    df['Potability'] = np.where(condition, 1, 0) # Restore target after masking
    df['Potability'] = np.where(mask, 1 - df['Potability'], df['Potability']) # Restore noise
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Dummy dataset created at {output_path}")

if __name__ == "__main__":
    create_dummy_data("data/raw/water_potability.csv")
