import joblib
import os
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import numpy as np
from datetime import datetime

ORIGINAL_MODEL_PATH = "models/yieldclassifier_model.pkl"
RETRAINED_MODELS_DIR = "retrainedmodels"

def load_model():
    if not os.path.exists(ORIGINAL_MODEL_PATH):
        raise FileNotFoundError("Original model file 'yieldclassifier_model.pkl' not found!")
    return joblib.load(ORIGINAL_MODEL_PATH)

def retrain_model(model, X_scaled, y):
    if not os.path.exists(RETRAINED_MODELS_DIR):
        os.makedirs(RETRAINED_MODELS_DIR)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    retrained_model_path = os.path.join(RETRAINED_MODELS_DIR, f"yieldclassifier_model_{timestamp}.pkl")
    
    model.fit(X_scaled, y)
    joblib.dump(model, retrained_model_path)
    return model, retrained_model_path

def predict(model, X_scaled):
    return model.predict(X_scaled)[0]

def evaluate_model(model, X_test_scaled, y_test):
    y_pred = model.predict(X_test_scaled)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred, average='macro'),
        "precision": precision_score(y_test, y_pred, average='macro'),
        "recall": recall_score(y_test, y_pred, average='macro'),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
    }
    return metrics