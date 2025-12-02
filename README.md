# 🫀 ECG Classification - Détection de Maladies Cardiovasculaires

## 📌 Description
Système intelligent de détection automatique d'anomalies cardiaques à partir d'images d'électrocardiogrammes (ECG) utilisant le Deep Learning.

## 🎯 Objectifs
- Classifier automatiquement les ECG en plusieurs catégories de pathologies
- Aider les professionnels de santé dans le diagnostic rapide et précis
- Déployer un système complet (Modèle + API + Interface)

## 🏗️ Architecture du Projet
```
ecg-classification/
├── src/              # Code source principal
├── notebooks/        # Notebooks d'exploration
├── tests/           # Tests unitaires
├── data/            # Données (non versionnées)
├── models/          # Modèles entraînés
├── streamlit_app/   # Interface utilisateur
└── docker/          # Configuration Docker
```

## 🚀 Installation

### Prérequis
- Python 3.10+
- Anaconda ou Miniconda

### Installation
```bash
# Créer l'environnement
conda create -n ecg_project python=3.10 -y
conda activate ecg_project

# Installer les dépendances
pip install -r requirements.txt
```

## 📊 Dataset
- Source: [À compléter]
- Nombre d'images: [À compléter]
- Classes: [À compléter]

## 🧪 Utilisation

### Exploration des données
```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

### Entraînement
```bash
python src/models/training.py
```

### Lancement de l'API
```bash
uvicorn src.api.main:app --reload
```

### Interface Streamlit
```bash
streamlit run streamlit_app/app.py
```

### MLflow UI
```bash
mlflow ui
```

## 👥 Équipe
- [Votre nom]
- [Autres membres si groupe]

## 📅 Timeline
- Semaine 1: Setup & Exploration
- Semaine 2: Preprocessing & Baseline
- Semaine 3: Training & Optimization
- Semaine 4: Expérimentations avancées
- Semaine 5: Déploiement (API + UI)
- Semaine 6: Tests & Documentation

## 📧 Contact
- Email: [votre email]
- Enseignant: Haythem Ghazouani (h.ghazouani@pi.tn)

## 📝 Licence
Projet académique - Terminale Data Science