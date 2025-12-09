import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# --- Définition des chemins ---
BASE_DIR = os.path.dirname(os.path.dirname(__file__))       # racine projet
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed") # data/processed

print("Loading dataset from:", PROCESSED_DIR)

# --- Charger données ---
images = np.load(os.path.join(PROCESSED_DIR, "images.npy"))
labels = np.load(os.path.join(PROCESSED_DIR, "labels.npy"))

# normalisation
images = images / 255.0

# --- Construction modèle CNN ---
model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(128, 128, 1)),
    MaxPooling2D((2, 2)),

    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),

    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.3),
    Dense(1, activation="sigmoid")  # binaire: Normal / Abnormal
])

model.compile(optimizer="adam",
              loss="binary_crossentropy",
              metrics=["accuracy"])

# --- Entraînement ---
history = model.fit(
    images, labels,
    batch_size=32,
    epochs=10,
    validation_split=0.2
)

# --- Sauvegarde du modèle ---
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

model.save(os.path.join(MODEL_DIR, "ecg_model.h5"))

print("Training terminé ✔")
