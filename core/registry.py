import os

MODEL_REGISTRY = {}

def register_model(model_name, output_dir):
    model_path = os.path.join(output_dir, model_name)
    MODEL_REGISTRY[model_name] = model_path
    print(f"Modèle {model_name} enregistré à : {model_path}")
