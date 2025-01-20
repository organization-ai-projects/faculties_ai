import logging
from pathlib import Path

# Importation des fonctions et variables nécessaires
from core.config import HF_TOKEN, HYPERPARAMETERS
from core.utils.data_utils import preprocess_data, load_splits, load_dataset
from core.utils.model_utils import train_model
from core.registry import register_model

# Configuration de logging pour afficher les logs dans la console
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Vérifier si le dataset est prétraité, sinon le générer
    processed_path = Path(HYPERPARAMETERS["dataset_path"])
    if not processed_path.exists():
        logger.info(f"Le dataset {processed_path} est introuvable. Prétraitement en cours...")
        preprocess_data("data/raw_data", processed_path)

    # Charger les splits si disponibles
    split_dir = Path("data/splits")
    if split_dir.exists():
        logger.info("Chargement des splits (train/val/test)...")
        splits = load_splits(split_dir)
        dataset = splits.get("train", [])  # Par défaut, utiliser le split 'train'
    else:
        logger.warning("Aucun split trouvé. Chargement du dataset complet...")
        dataset = load_dataset(HYPERPARAMETERS["dataset_path"])

    # Vérifier que le dataset est non vide
    if not dataset:
        raise ValueError("Le dataset est vide. Vérifiez vos données.")
    
    # Entraîner chaque modèle
    for model_name in HYPERPARAMETERS["models"]:
        logger.info(f"\n=== Début de l'entraînement pour le modèle : {model_name} ===")
        train_model(model_name, HYPERPARAMETERS, dataset, HF_TOKEN)
        register_model(model_name, HYPERPARAMETERS["output_dir"])
        logger.info(f"=== Modèle {model_name} terminé et sauvegardé ===")

if __name__ == "__main__":
    main()
