# -----------------------------
# train_efficientnet.py (version optimisée)
# -----------------------------

# Importations
import numpy as np
import torch
import os
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
from torchvision import models, transforms
import torch.optim as optim
import mlflow
import mlflow.pytorch
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

# Fonction pour charger les données
def load_data(data_path, batch_size=32, train_split=0.8):
    """
    Charge les données ECG et crée les DataLoaders.
    Assume que les données sont stockées comme un dictionnaire avec 'signals' et 'labels'.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Fichier de données non trouvé: {data_path}")
    
    data = np.load(data_path, allow_pickle=True).item()
    signals = data['signals']  # Shape: (n_samples, n_timesteps)
    labels = data['labels']    # Shape: (n_samples,)
    
    # Convertir en tenseurs PyTorch
    signals = torch.tensor(signals, dtype=torch.float32).unsqueeze(1)  # Ajouter canal pour Conv2D, shape: (n, 1, timesteps)
    labels = torch.tensor(labels, dtype=torch.long)
    
    # Dupliquer le canal pour simuler 3 canaux si nécessaire, ou adapter pour EfficientNet
    signals = signals.repeat(1, 3, 1, 1)  # Pour simuler RGB, mais pour ECG, peut-être pas idéal, ajuster selon besoin
    
    dataset = TensorDataset(signals, labels)
    
    # Split train/val
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

# Fonction pour créer le modèle EfficientNet optimisé
def create_model(num_classes, pretrained=True):
    """
    Crée le modèle EfficientNet_B0 avec des modifications pour ECG.
    """
    if pretrained:
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    else:
        model = efficientnet_b0()
    
    # Adapter la première couche pour les dimensions ECG (assume timesteps comme hauteur, 1 comme largeur initialement)
    # EfficientNet attend 3 canaux, nous avons simulé 3, mais pour ECG 1D, peut-être mieux de traiter comme image 1D ou adapter.
    # Pour simplification, on garde, mais en pratique, adapter la conv1.
    model.features[0][0] = nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)  # Ajuster si nécessaire
    
    # Remplacer la couche classifier
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    
    return model

# Fonction d'entraînement optimisée
def train_model(model, train_loader, val_loader, num_epochs=10, learning_rate=0.001, device='cuda', experiment_name='efficientnet_training'):
    """
    Entraîne le modèle avec MLflow pour le suivi.
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # MLflow
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run():
        mlflow.log_param('model', 'EfficientNet_B0')
        mlflow.log_param('num_epochs', num_epochs)
        mlflow.log_param('learning_rate', learning_rate)
        mlflow.log_param('batch_size', train_loader.batch_size)
        
        best_val_acc = 0.0
        
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            train_loss = running_loss / len(train_loader)
            train_acc = 100 * correct / total
            
            # Validation
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            val_loss /= len(val_loader)
            val_acc = 100 * correct / total
            
            print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            
            # Log to MLflow
            mlflow.log_metric('train_loss', train_loss, step=epoch)
            mlflow.log_metric('train_accuracy', train_acc, step=epoch)
            mlflow.log_metric('val_loss', val_loss, step=epoch)
            mlflow.log_metric('val_accuracy', val_acc, step=epoch)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), 'best_efficientnet_model.pth')
                mlflow.pytorch.log_model(model, 'best_model')
        
        print('Entraînement terminé.')

# Fonction principale
def main(data_path, num_classes=5, num_epochs=10, batch_size=32):
    """
    Fonction principale pour entraîner EfficientNet sur les données ECG.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Utilisation du device: {device}")
    
    # Charger les données
    train_loader, val_loader = load_data(data_path, batch_size=batch_size)
    
    # Créer le modèle
    model = create_model(num_classes)
    
    # Entraîner le modèle
    train_model(model, train_loader, val_loader, num_epochs=num_epochs, device=device)

if __name__ == "__main__":
    # Exemple d'utilisation
    data_path = 'data/processed/ecg_data.npy'  # Ajuster selon le chemin réel
    main(data_path)