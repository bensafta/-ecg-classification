import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import mlflow.pytorch

# -----------------------------
# Config
# -----------------------------
IMAGE_SIZE = 224  # à adapter selon ton modèle
RUN_ID_VGG16 = "3e6251aeb5f24635a5d3241bef702fbd"
RUN_ID_EFFICIENTNET = "placeholder_run_id"  # TODO: Replace with actual MLflow run ID for EfficientNet

MODEL_PATHS = {
    "VGG16": f"mlruns/0/{RUN_ID_VGG16}/artifacts/vgg16_model/pytorch_model.bin",
    "EfficientNet": f"mlruns/0/{RUN_ID_EFFICIENTNET}/artifacts/efficientnet_model/pytorch_model.bin"
}


# -----------------------------
# Streamlit Interface
# -----------------------------
st.title("ECG Classification")

# 1️⃣ Choix du modèle
model_name = st.selectbox("Choisir le modèle :", ["VGG16", "EfficientNet"])

# 2️⃣ Upload image
uploaded_file = st.file_uploader("Upload une image ECG", type=["png", "jpg", "jpeg"])

# -----------------------------
# Prétraitement
# -----------------------------
def preprocess_image(image: Image.Image):
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)  # ajouter batch dim

# -----------------------------
# Charger modèle
# -----------------------------
def load_model(model_name):
    if model_name == "VGG16":
        model = models.vgg16(weights=None)
        model.classifier[6] = nn.Linear(4096, 4)  # 4 classes ECG
    elif model_name == "EfficientNet":
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 4)
    else:
        raise ValueError("Modèle non supporté")
    
    model.load_state_dict(torch.load(MODEL_PATHS[model_name], map_location="cpu"))
    model.eval()
    return model

# -----------------------------
# Prédiction
# -----------------------------
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    input_tensor = preprocess_image(image)
    
    model = load_model(model_name)
    
    with torch.no_grad():
        output = model(input_tensor)
        predicted_class = torch.argmax(output, dim=1).item()
    
    st.image(image, caption="Image uploadée", use_column_width=True)
    st.write(f"Prédiction : Classe {predicted_class}")

# -----------------------------
# Option MLflow Logging
# -----------------------------
if st.button("Log modèle dans MLflow"):
    model = load_model(model_name)
    example_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)

    with mlflow.start_run() as run:
        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path=f"{model_name}_model",
            input_example=example_input
        )
        st.success(f"Modèle {model_name} loggé dans MLflow Run ID: {run.info.run_id}")
