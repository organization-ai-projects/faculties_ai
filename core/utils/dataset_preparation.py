import logging
from pathlib import Path
from core.utils.data_utils import preprocess_data, load_dataset

logger = logging.getLogger(__name__)

def prepare_dataset(directory_path, dataset_path, tokenizer):
    """Prépare le dataset pour l'entraînement."""
    try:
        processed_path = Path(dataset_path)
        if not processed_path.exists():
            logger.info("Les données prétraitées sont introuvables. Prétraitement en cours...")
            preprocess_data(directory_path, processed_path)

        dataset = load_dataset(processed_path)
        if not dataset or len(dataset) == 0:
            raise ValueError("Le dataset est vide après le chargement.")

        train_dataset, val_dataset = dataset.train_test_split(test_size=0.2).values()

        def preprocess_function(examples):
            return tokenizer(
                examples["input"], truncation=True, padding="max_length", max_length=128
            )

        tokenized_train = train_dataset.map(preprocess_function, batched=True, remove_columns=train_dataset.column_names)
        tokenized_val = val_dataset.map(preprocess_function, batched=True, remove_columns=val_dataset.column_names)

        return tokenized_train, tokenized_val
    except Exception as e:
        logger.error(f"Erreur lors de la préparation des données : {e}")
        return None, None
