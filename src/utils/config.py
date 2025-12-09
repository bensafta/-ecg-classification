# src/utils/config.py

import os
import torch

# -------------------------
# Paramètres généraux
# -------------------------
ARCH = 'baseline'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 128       # doit correspondre à preprocessing.py
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 10

# -------------------------
# Répertoires
# -------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # src/
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

# -------------------------
# Fichier modèle
# -------------------------
MODEL_FILENAME = f"ecg_model_{ARCH}.pth"
MODEL_PATH = os.path.join(PROCESSED_DIR, MODEL_FILENAME)

# -------------------------
# Paramètres MLflow (optionnel)
# -------------------------
MLFLOW_EXPERIMENT = "ECG_Classification"
PATIENCE = 5  # early stopping
