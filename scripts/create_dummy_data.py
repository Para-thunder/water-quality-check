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
        'Potability': np.random.randint(0, 2, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Introduce some missing values
    mask = np.random.choice([True, False], size=df.shape, p=[0.1, 0.9])
    df = df.mask(mask)
    # Ensure target has no missing values
    df['Potability'] = data['Potability']
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Dummy dataset created at {output_path}")

if __name__ == "__main__":
    create_dummy_data("data/raw/water_potability.csv")
