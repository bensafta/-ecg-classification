# -------------------------
# preprocessing.py
# -------------------------
import os
import numpy as np
from PIL import Image

# -------------------------
# Définition des chemins
# -------------------------

# le chemin du fichier Python actuel processing
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
#C’est là que se trouvent les images originales (non traitées).
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
#les données prêtes pour l’entraînement du modèle.
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

# Créer le dossier processed s'il n'existe pas 
os.makedirs(PROCESSED_DIR, exist_ok=True)

# -------------------------
# Paramètres
# -------------------------
IMAGE_SIZE = 224 #Compatibilité avec les modèles pré-entraînés(EfficientNet,VGG16,ResNet),
#Temps d’entraînement raisonnable  pas besoin d’ultra haute résolution puisque dans 
#notre projet formes , les pices et la les variations de rythme)

# Classes utilisées deux categories normal et anormal 
categories = {
    "Normal": 0,
    "Abnormal_heartbeat": 1
}
#contient toutes les images ECG prétraitées, stockées dans un tableau NumPy
images = []
#labels.npy contient la classe associée à chaque image.
labels = []

# -------------------------
# Début du preprocessing
# -------------------------
print("Début du preprocessing (Normal / Abnormal_heartbeat)...")

for category, label_value in categories.items():
    category_path = os.path.join(RAW_DIR, category)

    if not os.path.isdir(category_path):
        print(f"ATTENTION : {category_path} n'est pas un dossier.")
        continue
#Affiche une information pour savoir quelle catégorie est en cours de traitement.
#Utile pour déboguer et suivre la progression du preprocessing.
    print(f"Traitement de la catégorie : {category} -> label {label_value}")
#Parcourt tous les fichiers dans le dossier de la catégorie
    for img_name in os.listdir(category_path):
        img_path = os.path.join(category_path, img_name)

        try:
            # ouvrir en niveau de gris
            #convertit en niveau de gris (1 canal au lieu de 3RBG).
             # car ECG = signal monochrome → inutile de garder 3 canaux couleur.
             # pour Réduit la mémoire et simplifie l’apprentissage.
            img = Image.open(img_path).convert("L")
            # redimensionner les image en 224*224
            img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
            # normalisation(Convertit l’image en tableau NumPy avec des valeurs entre 0 et 1
            #pour stabilise l'apprentissage)
            img_array = np.array(img) / 255.0
           #liste contenant tous les tableaux d’images.
            images.append(img_array)
           # liste contenant le label numérique correspondant.
            labels.append(label_value)

        except Exception as e:
            print(f"Erreur lecture image {img_name} : {e}")

# -------------------------
# Conversion en numpy arrays
# -------------------------
#reshape images.shape → (N, 224, 224, 1): Les CNN attendent souvent des tableaux 4D : (batch, height, width, channels)
images = np.array(images).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)
labels = np.array(labels)

# -------------------------
# Sauvegarde
# -------------------------
np.save(os.path.join(PROCESSED_DIR, "images.npy"), images)
np.save(os.path.join(PROCESSED_DIR, "labels.npy"), labels)
np.save(os.path.join(PROCESSED_DIR, "label_map.npy"), categories)

# -------------------------
# Fin
# -------------------------
print("Preprocessing terminé ✔")
print("Images :", images.shape)
print("Labels :", labels.shape)
print("Label map :", categories)
