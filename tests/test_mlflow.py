# test_mlflow.py
import mlflow
import mlflow.pytorch
import yaml

# Charger la config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Configurer MLflow
mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
mlflow.set_experiment(config['mlflow']['experiment_name'])

# Test d'un run simple
with mlflow.start_run(run_name="test_run"):
    # Logger des paramètres
    mlflow.log_param("test_param", "hello")
    mlflow.log_metric("test_metric", 0.95)
    
    print("✅ MLflow fonctionne correctement!")
    print(f"Run ID: {mlflow.active_run().info.run_id}")

print("\n🎯 Pour voir l'interface MLflow, lancez:")
print("mlflow ui")
print("Puis ouvrez: http://localhost:5000")