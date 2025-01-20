import logging
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer
from core import config
from core.utils.data_utils import preprocess_data, load_dataset
from pathlib import Path

# Configuration des logs
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/fine_tune.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def fine_tune_model(model_name, dataset_path, output_dir, hf_token):
    """Fine-tune un modèle pré-entraîné sur un dataset spécifique."""
    logger.info(f"Début du fine-tuning pour {model_name} sur {dataset_path}")

    # Charger le modèle et le tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=hf_token)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)

    # Prétraiter les données si nécessaire
    processed_path = Path(dataset_path)
    if not processed_path.exists():
        logger.error(f"Le fichier {dataset_path} est introuvable. Prétraitement requis.")
        return

    dataset = load_dataset(dataset_path)
    if not dataset:
        logger.error("Échec du chargement du dataset.")
        return

    # Diviser les données en ensembles d'entraînement et de validation
    train_dataset, val_dataset = dataset.train_test_split(test_size=0.2).values()

    def preprocess_function(examples):
        return tokenizer(examples["input"], truncation=True, padding="max_length", max_length=128)

    tokenized_train = train_dataset.map(preprocess_function, batched=True, remove_columns=train_dataset.column_names)
    tokenized_val = val_dataset.map(preprocess_function, batched=True, remove_columns=val_dataset.column_names)

    # Configurer l'entraînement
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,  # Fine-tuning sur 3 époques par défaut
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        logging_dir="logs/fine_tune",
        logging_steps=50,
        learning_rate=5e-5,
        load_best_model_at_end=True,
    )

    # Créer le trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val
    )

    # Lancer le fine-tuning
    trainer.train()
    logger.info(f"Fine-tuning terminé pour {model_name}. Modèle sauvegardé dans {output_dir}")

def main():
    try:
        hf_token = config.HF_TOKEN
        dataset_path = "data/processed/training_dataset.json"
        output_dir = "models/fine_tuned"

        for model_name in config.HYPERPARAMETERS["models"]:
            fine_tune_model(model_name, dataset_path, output_dir, hf_token)
    except Exception as e:
        logger.error(f"Erreur lors du fine-tuning : {e}")

if __name__ == "__main__":
    main()
