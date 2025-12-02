import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
import mlflow
import mlflow.pytorch

# -------------------------
# Chemins et hyperparamètres
# -------------------------
PROCESSED_DIR = 'data/processed'
MODEL_PATH = 'baseline_model.pth'

BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001
IMAGE_SIZE = 224

# -------------------------
# Charger les données
# -------------------------
images = np.load(f'{PROCESSED_DIR}/images.npy')
labels = np.load(f'{PROCESSED_DIR}/labels.npy')
images = images[:, np.newaxis, :, :]  # ajouter canal pour CNN

images_tensor = torch.tensor(images, dtype=torch.float32)
labels_tensor = torch.tensor(labels, dtype=torch.long)

dataset = TensorDataset(images_tensor, labels_tensor)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# -------------------------
# Définir le modèle CNN
# -------------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 56 * 56)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN(num_classes=4).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# -------------------------
# MLflow setup
# -------------------------
mlflow.set_experiment("ECG_Classification")

train_losses = []
train_accuracies = []

with mlflow.start_run():
    mlflow.log_param("epochs", EPOCHS)
    mlflow.log_param("batch_size", BATCH_SIZE)
    mlflow.log_param("learning_rate", LEARNING_RATE)

    # -------------------------
    # Boucle d'entraînement
    # -------------------------
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)

        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {epoch_loss:.4f} - Accuracy: {epoch_acc:.4f}")

        # Log metrics dans MLflow
        mlflow.log_metric("train_loss", epoch_loss, step=epoch)
        mlflow.log_metric("train_accuracy", epoch_acc, step=epoch)

    # -------------------------
    # Évaluation sur le test set
    # -------------------------
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    test_acc = correct / total
    print(f"Test Accuracy: {test_acc:.4f}")
    mlflow.log_metric("test_accuracy", test_acc)

    # -------------------------
    # Sauvegarder le modèle
    # -------------------------
    torch.save(model.state_dict(), MODEL_PATH)
    mlflow.pytorch.log_model(model, "baseline_model")
    print(f"Modèle sauvegardé sous {MODEL_PATH}")

# -------------------------
# Courbes Loss & Accuracy
# -------------------------
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(range(1,EPOCHS+1), train_losses, marker='o')
plt.title("Train Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.subplot(1,2,2)
plt.plot(range(1,EPOCHS+1), train_accuracies, marker='o', color='orange')
plt.title("Train Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.show()

# -------------------------
# Fonction pour tester une nouvelle image
# -------------------------
def predict_image(img_path):
    model.eval()
    img = Image.open(img_path).convert('L').resize((IMAGE_SIZE, IMAGE_SIZE))
    img_array = np.array(img) / 255.0
    img_tensor = torch.tensor(img_array[np.newaxis, np.newaxis, :, :], dtype=torch.float32).to(device)
    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)
    print(f"Prediction pour {img_path}: Classe {predicted.item()}")
