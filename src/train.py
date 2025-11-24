import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, Flatten, MaxPooling1D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import mlflow
import mlflow.tensorflow
import mlflow.sklearn
import os
import sys
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils import class_weight
from sklearn.ensemble import RandomForestClassifier

def load_data(data_dir):
    train_path = os.path.join(data_dir, 'train.csv')
    test_path = os.path.join(data_dir, 'test.csv')
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    X_train = train_df.drop('Potability', axis=1)
    y_train = train_df['Potability']
    X_test = test_df.drop('Potability', axis=1)
    y_test = test_df['Potability']
    
    return X_train, y_train, X_test, y_test

def build_ann_model(input_dim):
    model = Sequential([
        Dense(256, activation='relu', input_dim=input_dim),
        BatchNormalization(),
        Dropout(0.3),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_cnn_model(input_dim):
    model = Sequential([
        Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(input_dim, 1)),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_and_evaluate(data_dir, model_type='ann', epochs=50, batch_size=32):
    X_train, y_train, X_test, y_test = load_data(data_dir)
    
    input_dim = X_train.shape[1]
    
    if model_type == 'cnn':
        # Reshape for CNN: (samples, features, 1)
        X_train_reshaped = X_train.values.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test_reshaped = X_test.values.reshape(X_test.shape[0], X_test.shape[1], 1)
        model = build_cnn_model(input_dim)
        X_train_fit = X_train_reshaped
        X_test_eval = X_test_reshaped
    elif model_type == 'rf':
        model = RandomForestClassifier(n_estimators=500, max_depth=None, random_state=42, class_weight='balanced')
        X_train_fit = X_train
        X_test_eval = X_test
    else:
        model = build_ann_model(input_dim)
        X_train_fit = X_train
        X_test_eval = X_test

    mlflow.set_experiment("Water_Potability_Prediction")
    
    with mlflow.start_run(run_name=f"{model_type.upper()}_Model"):
        mlflow.log_param("model_type", model_type)
        
        if model_type == 'rf':
            print(f"Training {model_type.upper()} model...")
            model.fit(X_train_fit, y_train)
            
            print("Evaluating model...")
            # RF specific prediction
            y_pred = model.predict(X_test_eval)
            y_pred_prob = model.predict_proba(X_test_eval)[:, 1]
            
            accuracy = accuracy_score(y_test, y_pred)
            
            # Save RF model
            os.makedirs("models", exist_ok=True)
            model_path = os.path.join("models", f"{model_type}_model.pkl")
            joblib.dump(model, model_path)
            print(f"Model saved to {model_path}")
            
            mlflow.sklearn.log_model(model, "model")
            
        else:
            mlflow.log_param("epochs", epochs)
            mlflow.log_param("batch_size", batch_size)
            
            print(f"Training {model_type.upper()} model...")
            
            # Calculate class weights to handle imbalance
            class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
            class_weight_dict = dict(enumerate(class_weights))
            print(f"Class weights: {class_weight_dict}")
            
            # Callbacks for better training
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001, verbose=1)
            # Increased patience to allow model to learn longer before stopping
            early_stop = EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True, verbose=1)
            
            history = model.fit(X_train_fit, y_train, epochs=epochs, batch_size=batch_size, 
                              validation_split=0.2, verbose=2, class_weight=class_weight_dict,
                              callbacks=[reduce_lr, early_stop])
            
            print("Evaluating model...")
            loss, accuracy = model.evaluate(X_test_eval, y_test, verbose=0)
            
            y_pred_prob = model.predict(X_test_eval)
            y_pred = (y_pred_prob > 0.5).astype(int)
            
            # Save Keras model
            os.makedirs("models", exist_ok=True)
            model_path = os.path.join("models", f"{model_type}_model.keras")
            model.save(model_path)
            print(f"Model saved to {model_path}")
            
            mlflow.tensorflow.log_model(model, "model")
        
        # Common metrics calculation
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        mlflow.log_metric("accuracy", float(accuracy))
        mlflow.log_metric("precision", float(precision))
        mlflow.log_metric("recall", float(recall))
        mlflow.log_metric("f1_score", float(f1))
        
        # Save metrics to JSON for DVC
        import json
        metrics = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1)
        }
        with open("metrics.json", "w") as f:
            json.dump(metrics, f)
        
        print(f"{model_type.upper()} Model - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/train.py <data_dir> [model_type]")
        sys.exit(1)
        
    data_dir = sys.argv[1]
    model_type = sys.argv[2] if len(sys.argv) > 2 else 'ann'
    
    train_and_evaluate(data_dir, model_type)
