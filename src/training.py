# training.py
import os
from nbconvert import NotebookExporter
from nbclient import NotebookClient
from nbformat import read, write, NO_CONVERT


# -------------------------
# Chemins des notebooks
# -------------------------
SRC = [
    "src/train/train_CNN.ipynb",
    "src/train/train_EfficientNet.ipynb",
    "src/train/train_ResNet.ipynb",
    "src/train/train_VGG16.ipynb"
   
]

# -------------------------
# Fonction pour exécuter un notebook
# -------------------------
def run_notebook(nb_path):
    print(f"\n=== Lancement du src : {nb_path} ===")
    
    # Ouvrir le notebook
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = read(f, as_version=4)
    
    # Créer le client pour exécuter les cellules
    client = NotebookClient(nb, timeout=3600, kernel_name="python3")
    client.execute()
    
    # Optionnel : sauvegarder le notebook exécuté
    executed_path = nb_path.replace(".ipynb", "_executed.ipynb")
    with open(executed_path, "w", encoding="utf-8") as f:
        write(nb, f)
    
    print(f"✅ Notebook terminé et sauvegardé : {executed_path}")

# -------------------------
# Boucle sur tous les notebooks
# -------------------------
for nb_file in SRC:
    if os.path.exists(nb_file):
        run_notebook(nb_file)
    else:
        print(f"❌ Notebook introuvable : {nb_file}")
