# Water Potability MLOps Project

This project implements an end-to-end MLOps pipeline for predicting water potability using Deep Learning (ANN & CNN). It integrates **DVC** for data versioning, **MLflow** for experiment tracking, **Docker** for containerization, and **Airflow** for orchestration.

## Project Structure

- `src/` - Source code for data preparation and training.
- `data/` - Data directory (managed by DVC).
- `notebooks/` - Jupyter notebooks for exploration.
- `tests/` - Unit tests.
- `scripts/` - Automation scripts.
- `config/` - Configuration files.
- `dvc.yaml` - DVC pipeline definition.

## Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Data Setup**:
   - Place the `water_potability.csv` dataset in `data/raw/`.
   - You can download it from [Kaggle](https://www.kaggle.com/adityakadiwal/water-potability).

3. **Run Pipeline**:
   - Execute the DVC pipeline:
     ```bash
     dvc repro
     ```

4. **MLflow**:
   - View experiments:
     ```bash
     mlflow ui
     ```

## Components

- **Data Versioning**: DVC
- **Experiment Tracking**: MLflow
- **Modeling**: TensorFlow/Keras (ANN, CNN)
- **Orchestration**: Apache Airflow (Coming soon)
- **Deployment**: Docker, AWS (Coming soon)

## License

MIT
