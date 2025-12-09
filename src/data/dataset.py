# src/data/dataset.py
import os
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
from ..utils.config import PROCESSED_DIR, BATCH_SIZE, DEVICE

def get_data_loaders(batch_size=BATCH_SIZE, test_split=0.2):
    images = np.load(os.path.join(PROCESSED_DIR, "images.npy"))
    labels = np.load(os.path.join(PROCESSED_DIR, "labels.npy"))
    
    # Ajouter dimension canal si manquant (N,H,W) -> (N,C,H,W)
    if images.ndim == 3:
        images = images[:, np.newaxis, :, :]
    
    images_tensor = torch.tensor(images, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    
    dataset = TensorDataset(images_tensor, labels_tensor)
    test_size = int(test_split * len(dataset))
    train_size = len(dataset) - test_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

