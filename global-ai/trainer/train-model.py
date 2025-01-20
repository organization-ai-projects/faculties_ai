from core.config import HF_TOKEN, HYPERPARAMETERS
from core.utils.data_utils import load_dataset, preprocess_data
from core.utils.model_utils import train_model
from core.registry import register_model

def main():
    # Charger les données
    dataset = load_dataset(HYPERPARAMETERS["dataset_path"])

    # Entraîner chaque modèle spécifié
    for model_name in HYPERPARAMETERS["models"]:
        print(f"Entraînement du modèle : {model_name}")
        train_model(model_name, HYPERPARAMETERS, dataset, HF_TOKEN)
        register_model(model_name, HYPERPARAMETERS["output_dir"])

if __name__ == "__main__":
    main()
