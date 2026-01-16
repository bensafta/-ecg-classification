# -------------------------
# data_exploration.py (version optimisée)
# -------------------------

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Fonction optimisée pour charger les données ECG
def load_ecg_data(data_path):
    """
    Charge les données ECG depuis un fichier numpy.
    Assume que les données sont stockées comme un dictionnaire avec 'signals' et 'labels'.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Fichier de données non trouvé: {data_path}")
    
    data = np.load(data_path, allow_pickle=True).item()
    signals = data['signals']  # Shape: (n_samples, n_timesteps)
    labels = data['labels']    # Shape: (n_samples,)
    return signals, labels

# Fonction pour visualiser un échantillon d'ECG
def plot_ecg_sample(signals, labels, sample_idx=0, save_path=None):
    """
    Affiche un échantillon d'ECG.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(signals[sample_idx])
    plt.title(f'Échantillon ECG - Classe: {labels[sample_idx]}')
    plt.xlabel('Temps')
    plt.ylabel('Amplitude')
    if save_path:
        plt.savefig(save_path)
    plt.show()

# Fonction pour calculer les statistiques de base
def compute_statistics(signals):
    """
    Calcule les statistiques de base des signaux ECG.
    """
    mean_signal = np.mean(signals, axis=0)
    std_signal = np.std(signals, axis=0)
    min_signal = np.min(signals, axis=0)
    max_signal = np.max(signals, axis=0)
    return mean_signal, std_signal, min_signal, max_signal

# Fonction pour la réduction de dimension avec PCA
def apply_pca(signals, n_components=2):
    """
    Applique PCA pour réduire la dimension des signaux.
    """
    # Aplatir les signaux pour PCA
    signals_flat = signals.reshape(signals.shape[0], -1)
    pca = PCA(n_components=n_components)
    signals_pca = pca.fit_transform(signals_flat)
    return signals_pca, pca

# Fonction pour la réduction de dimension avec t-SNE (optimisée)
def apply_tsne(signals, n_components=2, perplexity=30, n_iter=1000):
    """
    Applique t-SNE pour réduire la dimension des signaux.
    Utilise moins d'itérations pour optimisation.
    """
    signals_flat = signals.reshape(signals.shape[0], -1)
    tsne = TSNE(n_components=n_components, perplexity=perplexity, n_iter=n_iter, random_state=42)
    signals_tsne = tsne.fit_transform(signals_flat)
    return signals_tsne

# Fonction pour visualiser les clusters avec PCA ou t-SNE
def plot_reduced_data(reduced_data, labels, method='PCA', save_path=None):
    """
    Visualise les données réduites.
    """
    plt.figure(figsize=(8, 6))
    unique_labels = np.unique(labels)
    for label in unique_labels:
        mask = labels == label
        plt.scatter(reduced_data[mask, 0], reduced_data[mask, 1], label=f'Classe {label}', alpha=0.7)
    plt.title(f'Visualisation avec {method}')
    plt.xlabel(f'{method} 1')
    plt.ylabel(f'{method} 2')
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.show()

# Fonction principale pour l'exploration des données
def main(data_path, output_dir='outputs'):
    """
    Fonction principale pour explorer les données ECG.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Charger les données
    print("Chargement des données...")
    signals, labels = load_ecg_data(data_path)
    print(f"Données chargées: {signals.shape[0]} échantillons, {signals.shape[1]} pas de temps")
    
    # Statistiques
    print("Calcul des statistiques...")
    mean_sig, std_sig, min_sig, max_sig = compute_statistics(signals)
    
    # Visualiser un échantillon
    plot_ecg_sample(signals, labels, sample_idx=0, save_path=os.path.join(output_dir, 'sample_ecg.png'))
    
    # PCA
    print("Application de PCA...")
    signals_pca, pca = apply_pca(signals, n_components=2)
    plot_reduced_data(signals_pca, labels, method='PCA', save_path=os.path.join(output_dir, 'pca_visualization.png'))
    
    # t-SNE (optimisé avec moins d'itérations)
    print("Application de t-SNE...")
    signals_tsne = apply_tsne(signals, n_components=2, n_iter=500)  # Réduit pour optimisation
    plot_reduced_data(signals_tsne, labels, method='t-SNE', save_path=os.path.join(output_dir, 'tsne_visualization.png'))
    
    print("Exploration terminée. Résultats sauvegardés dans", output_dir)

if __name__ == "__main__":
    # Exemple d'utilisation
    data_path = 'data/processed/ecg_data.npy'  # Ajuster selon le chemin réel
    main(data_path)