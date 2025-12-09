# src/api/endpoints.py

import os
import torch
from fastapi import APIRouter, UploadFile, File
from PIL import Image
import numpy as np

# Import du modèle et configuration
from src.models.architecture import SimpleCNN
from src.utils.config import DEVICE, ARCH, MODEL_PATH, PROCESSED_DIR, IMAGE_SIZE

# -------------------------
# Vérification du modèle
# -------------------------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Le fichier du modèle est introuvable : {MODEL_PATH}\n"
        "Veuillez exécuter training.py pour générer le modèle."
    )

# -------------------------
# Initialisation du modèle
# -------------------------
model = SimpleCNN(num_classes=4, arch=ARCH).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print(f"Modèle chargé avec succès ✅ : {MODEL_PATH}")

# -------------------------
# Création du router FastAPI
# -------------------------
router = APIRouter()

# Classes des labels
CLASSES = ['Normal', 'MI_patients', 'Abnormal_heartbeat', 'History_MI']

# -------------------------
# Endpoint pour la prédiction
# -------------------------
@router.post("/predict")
async def predict_ecg(file: UploadFile = File(...)):
    # Vérification du type de fichier
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        return {"error": "Seuls les fichiers PNG/JPG sont supportés"}

    # Lecture et prétraitement de l'image
    image = Image.open(file.file).convert('L')
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    image_array = np.array(image) / 255.0
    image_tensor = torch.tensor(image_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)

    # Prédiction
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        predicted_class = CLASSES[predicted.item()]

    return {"filename": file.filename, "prediction": predicted_class}
