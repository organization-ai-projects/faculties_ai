import logging
import torch
from core import config
from core.utils.dataset_preparation import prepare_dataset
from core.utils.model_trainer import train_model
from core.utils.model_utils import load_model_and_tokenizer
from core.utils.training_args import configure_training_args

# Configuration des logs
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Pipeline principal pour entraîner tous les modèles définis dans les hyperparamètres."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Appareil détecté : {device}")

    for model_name in config.HYPERPARAMETERS["models"]:
        try:
            logger.info(f"Préparation de l'entraînement pour {model_name}...")

            # Charger le modèle et le tokenizer
            model, tokenizer = load_model_and_tokenizer(model_name, config.HF_TOKEN)
            if not model or not tokenizer:
                raise ValueError(f"Erreur de chargement pour le modèle {model_name}.")

            # Préparer les données
            train_dataset, val_dataset = prepare_dataset(
                directory_path="data/raw_data",
                dataset_path=config.HYPERPARAMETERS["dataset_path"],
                tokenizer=tokenizer
            )
            if not train_dataset or not val_dataset:
                raise ValueError(f"Erreur de préparation des datasets pour {model_name}.")

            # Configurer les arguments d'entraînement
            training_args = configure_training_args(model_name, config.HYPERPARAMETERS)

            # Entraîner le modèle
            train_model(
                model_name=model_name,
                hyperparameters=config.HYPERPARAMETERS,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                hf_token=config.HF_TOKEN,
                training_args=training_args,
                device=device
            )
        except Exception as e:
            logger.error(f"Erreur lors de l'entraînement pour {model_name} : {e}")

if __name__ == "__main__":
    main()
