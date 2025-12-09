import os
import cv2
import numpy as np

# --- Définition des chemins ---
BASE_DIR = os.path.dirname(os.path.dirname(__file__))       # racine du projet
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")             # data/raw
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed") # data/processed

print("RAW_DIR =", RAW_DIR)
print("PROCESSED_DIR =", PROCESSED_DIR)

# Assurer que processed existe
os.makedirs(PROCESSED_DIR, exist_ok=True)

# --- Paramètres du dataset ---
IMG_SIZE = 128

images = []
labels = []

label_map = {}   # ex: {"Normal": 0, "Abnormal": 1}
current_label = 0

# --- Lecture des classes (Normal / Abnormal) ---
for category in os.listdir(RAW_DIR):
    category_path = os.path.join(RAW_DIR, category)

    if not os.path.isdir(category_path):
        continue

    print(f"Processing category: {category_path}")

    # enregistrer nom->numéro
    label_map[category] = current_label
    label_value = current_label
    current_label += 1

    # --- Lecture des images ---
    for img_name in os.listdir(category_path):
        img_path = os.path.join(category_path, img_name)

        # lire image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"WARNING: cannot read {img_path}")
            continue

        # resize
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        images.append(img)
        labels.append(label_value)

# convertir en arrays
images = np.array(images).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
labels = np.array(labels)

# --- Sauvegarde ---
np.save(os.path.join(PROCESSED_DIR, "images.npy"), images)
np.save(os.path.join(PROCESSED_DIR, "labels.npy"), labels)
np.save(os.path.join(PROCESSED_DIR, "label_map.npy"), label_map)

print("Preprocessing terminé ✔")
print("Images:", images.shape)
print("Labels:", labels.shape)
print("Label map:", label_map)
