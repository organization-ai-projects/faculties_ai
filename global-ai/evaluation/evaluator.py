import os
from global_ai.evaluation.metrics import calculate_metrics

def evaluate_model(model_name, model_dir):
    """
    Évalue un modèle en utilisant les métriques définies.
    """
    model_path = os.path.join(model_dir, model_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Le modèle {model_name} est introuvable dans {model_path}.")

    # Charger le modèle et exécuter l'évaluation (simulé ici)
    print(f"Chargement du modèle depuis {model_path}...")
    # Implémente une logique d'évaluation, par exemple :
    results = calculate_metrics(model_path)
    return results
