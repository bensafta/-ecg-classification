import os
import numpy as np
from PIL import Image

# Chemins
RAW_DIR = 'data/raw'
PROCESSED_DIR = 'data/processed'

# Créer le dossier processed s'il n'existe pas
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Définir les catégories et labels
categories = {
    'Normal': 0,
    'MI_patients': 1,
    'Abnormal_heartbeat': 2,
    'History_MI': 3
}

images = []
labels = []

for category, label in categories.items():
    path = os.path.join(RAW_DIR, category)
    for img_name in os.listdir(path):
        img_path = os.path.join(path, img_name)
        img = Image.open(img_path).convert('L')  # convertir en grayscale si nécessaire
        img = img.resize((224, 224))  # redimensionner si besoin
        img_array = np.array(img) / 255.0  # normalisation
        images.append(img_array)
        labels.append(label)

# Convertir en numpy array
images = np.array(images)
labels = np.array(labels)

# Sauvegarder
np.save(os.path.join(PROCESSED_DIR, 'images.npy'), images)
np.save(os.path.join(PROCESSED_DIR, 'labels.npy'), labels)

print("Préprocessing terminé. Images et labels sauvegardés dans data/processed.")
