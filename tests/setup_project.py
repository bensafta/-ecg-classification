# setup_project.py
import os
from pathlib import Path

# Structure du projet
structure = {
    "src": {
        "data": ["__init__.py", "dataset.py", "preprocessing.py"],
        "models": ["__init__.py", "architecture.py", "training.py"],
        "api": ["__init__.py", "main.py", "endpoints.py"],
        "utils": ["__init__.py", "config.py", "metrics.py"]
    },
    "streamlit_app": {
        "components": ["__init__.py"]
    },
    "notebooks": [],
    "tests": ["test_data.py", "test_models.py", "test_api.py"],
    "docker": [],
    "data": {
        "raw": [],
        "processed": [],
        "splits": []
    },
    "models": []
}

def create_structure(base_path, structure):
    """Crée récursivement la structure de dossiers et fichiers"""
    for name, content in structure.items():
        path = Path(base_path) / name
        path.mkdir(parents=True, exist_ok=True)
        
        if isinstance(content, dict):
            create_structure(path, content)
        elif isinstance(content, list):
            for file in content:
                (path / file).touch()

# Créer la structure
create_structure(".", structure)

# Créer les fichiers racine
files_racine = [
    "requirements.txt",
    "README.md",
    ".gitignore",
    "config.yaml"
]

for file in files_racine:
    Path(file).touch()

# Créer les notebooks
notebooks = [
    "notebooks/01_data_exploration.ipynb",
    "notebooks/02_model_training.ipynb",
    "notebooks/03_evaluation.ipynb"
]

for notebook in notebooks:
    Path(notebook).touch()

# Créer fichiers dans streamlit_app
Path("streamlit_app/app.py").touch()

# Créer fichiers dans docker
Path("docker/Dockerfile.api").touch()
Path("docker/docker-compose.yml").touch()

print("✅ Structure du projet créée avec succès!")
print("\nArborescence créée:")
print("""
ecg-classification/
├── src/
│   ├── data/
│   ├── models/
│   ├── api/
│   └── utils/
├── streamlit_app/
├── notebooks/
├── tests/
├── docker/
├── data/
├── models/
└── fichiers de configuration
""")