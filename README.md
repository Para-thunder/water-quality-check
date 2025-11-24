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
- **API**: FastAPI
- **Interface**: Streamlit & Gradio
- **Containerization**: Docker
- **Orchestration**: Apache Airflow (DAGs provided)
- **Deployment**: AWS App Runner (Guide in `deployment/`)

## Running the App

1. **Start the API (Docker)**:
   ```bash
   docker start water-app
   ```
   Or build and run:
   ```bash
   docker build -t water-potability-app .
   docker run -d -p 8000:8000 --name water-app water-potability-app
   ```

2. **Start the Interface**:
   ```bash
   streamlit run src/streamlit_app.py
   ```

3. **Access**:
   - API Docs: http://localhost:8000/docs
   - Interface: http://localhost:8501

MIT
