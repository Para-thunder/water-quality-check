PS C:\Users\esaar\mlops_project> python src\predict_api_flask.py
2025-11-24 17:05:20.494543: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2025-11-24 17:05:29.994386: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
2025-11-24 17:05:40.183187: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: SSE3 SSE4.1 SSE4.2 AVX AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
Model loaded from models/ann_model.keras
Scaler loaded from models/scaler.pkl
Model input_shape: (None, 9)
Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape            ┃       Param # ┃    
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩    
│ dense (Dense)                   │ (None, 64)              │           640 │    
├─────────────────────────────────┼─────────────────────────┼───────────────┤    
│ dropout (Dropout)               │ (None, 64)              │             0 │    
├─────────────────────────────────┼─────────────────────────┼───────────────┤    
│ dense_1 (Dense)                 │ (None, 32)              │         2,080 │    
├─────────────────────────────────┼─────────────────────────┼───────────────┤    
│ dropout_1 (Dropout)             │ (None, 32)              │             0 │    
├─────────────────────────────────┼─────────────────────────┼───────────────┤    
│ dense_2 (Dense)                 │ (None, 1)               │            33 │    
└─────────────────────────────────┴─────────────────────────┴───────────────┘    
 Total params: 8,261 (32.27 KB)
 Trainable params: 2,753 (10.75 KB)
 Non-trainable params: 0 (0.00 B)
 Optimizer params: 5,508 (21.52 KB)
Scaler mean: [6.57432901e+00 1.99666681e+02 2.77325929e+04 7.30576077e+00
 3.49978922e+02 4.70818297e+02 1.30474105e+01 6.30466102e+01
 3.79262261e+00]
Scaler scale: [3.97740077e+00 5.47407041e+01 1.26003867e+04 4.18823812e+00       
 9.03293549e+01 1.38702666e+02 7.41639254e+00 3.24773266e+01
 1.64253566e+00]
 * Serving Flask app 'predict_api_flask'
 * Debug mode: off
WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
 * Running on all addresses (0.0.0.0)
 * Running on http://127.0.0.1:8000
 * Running on http://192.168.100.8:8000
Press CTRL+C to quit
127.0.0.1 - - [24/Nov/2025 17:05:44] "GET / HTTP/1.1" 200 -
127.0.0.1 - - [24/Nov/2025 17:05:44] "GET /favicon.ico HTTP/1.1" 404 -
[UI] Input features: [7.0, 200.0, 20000.0, 7.0, 300.0, 400.0, 15.0, 60.0, 4.0]
C:\Users\esaar\AppData\Local\Programs\Python\Python311\Lib\site-packages\sklearn\utils\validation.py:2749: UserWarning: X does not have valid feature names, but StandardScaler was fitted with feature names
  warnings.warn(
[UI] Applied scaler to input data
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 341ms/step
[UI] Prediction probability: 0.3453991711139679
127.0.0.1 - - [24/Nov/2025 17:05:47] "POST /predict_ui HTTP/1.1" 200 -
# Project Progress Report: Water Potability Prediction using MLOps

**Date:** November 24, 2025
**Deadline:** November 25, 2025

## 1. Group Members
*   [Member Name 1] - [ID]
*   [Member Name 2] - [ID]
*   [Member Name 3] - [ID]

## 2. Abstract
This project aims to develop a robust Machine Learning Operations (MLOps) pipeline for predicting water potability. By leveraging Deep Learning models (Artificial Neural Networks and Convolutional Neural Networks), we analyze water quality metrics to determine safety for consumption. The project integrates industry-standard tools such as DVC for data versioning, MLflow for experiment tracking, Docker for containerization, and Flask for model serving, ensuring a reproducible and scalable machine learning workflow.

## 3. Introduction
Access to safe drinking water is essential for health. Traditional methods of water quality testing can be labor-intensive. This project automates the assessment of water potability using machine learning. We utilize the "Water Potability" dataset to train models that classify water samples as potable or not based on chemical properties like pH, Hardness, Solids, and Chloramines.

The core objective is not just model development but the implementation of a complete MLOps lifecycle. This includes:
*   Automated data pipelines.
*   Experiment tracking to compare model performance.
*   Containerized deployment for consistency across environments.
*   User-friendly web interfaces for real-time predictions.

## 4. Proposed Methodology
Our methodology follows a structured MLOps pipeline:

### 4.1. Data Management
*   **Source:** Water Potability dataset (Kaggle).
*   **Versioning:** We use **DVC (Data Version Control)** to track changes in the dataset (`data/raw` vs `data/processed`) and manage the data pipeline stages defined in `dvc.yaml`.

### 4.2. Model Development
We implemented two Deep Learning architectures using **TensorFlow/Keras**:
*   **ANN (Artificial Neural Network):** A dense network with dropout layers for classification.
*   **CNN (Convolutional Neural Network):** A 1D convolutional network to capture patterns in the feature space.

### 4.3. Experiment Tracking
**MLflow** is used to log parameters (epochs, batch size), metrics (accuracy, loss), and artifacts (saved models). This allows us to compare different runs and select the best-performing model.

### 4.4. Deployment & Interface
*   **API:** A **Flask** application (`src/predict_api_flask.py`) serves the model, providing a REST API for predictions.
*   **Containerization:** A `Dockerfile` is created to package the application, dependencies, and model into a portable container.
*   **Orchestration:** **Apache Airflow** DAGs (`dags/water_potability_dag.py`) are designed to automate the training and evaluation workflows.

## 5. Initial Results
We have successfully set up the pipeline and executed initial training runs.

*   **Experiment Tracking:** Multiple runs have been logged in MLflow for both ANN and CNN models.
*   **Performance:** Initial training shows promising accuracy.
    *   *Current Best Accuracy:* ~91.4% (based on latest metrics).
*   **Visualizations:** We have generated confusion matrices and loss curves (tracked in MLflow artifacts).

*(Note: Precision and Recall metrics are currently being optimized to handle class imbalance effectively.)*

## 6. Code Structure
The project is organized as follows:

*   `src/`: Contains source code for training (`train.py`), data prep (`data_prep.py`), and the Flask app (`predict_api_flask.py`).
*   `models/`: Stores trained model files (`.keras`).
*   `mlruns/`: Local storage for MLflow tracking data.
*   `Dockerfile`: Configuration for building the Docker image.
*   `dvc.yaml`: Defines the data processing and training pipeline stages.
*   `requirements.txt`: Lists all Python dependencies.

## 7. Video Demonstration
A video demonstration of the project is attached separately. It covers:
1.  **Project Setup:** Running `dvc repro` and installing dependencies.
2.  **MLflow UI:** Showing tracked experiments and metrics.
3.  **Model Training:** Executing the training script.
4.  **Web Interface:** Launching the Flask app and making a prediction via the UI.
5.  **Docker:** Running the application inside a Docker container.
