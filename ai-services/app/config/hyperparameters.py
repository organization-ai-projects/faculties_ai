import os
from dotenv import load_dotenv

# Charger les variables d'environnement
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(dotenv_path=os.path.join(BASE_DIR, "../../.env"))

# Vérification des variables critiques
required_keys = ["DATASET_PATH", "LOGGING_DIR"]
for key in required_keys:
    if not os.getenv(key):
        raise ValueError(f"Missing environment variable: {key}")

# Création des répertoires communs
os.makedirs(
    os.getenv("LOGGING_DIR", os.path.join(BASE_DIR, "../../logs/")), exist_ok=True
)

# Environnement courant
APP_ENV = os.getenv("APP_ENV", "dev")

# Hyperparamètres communs
HYPERPARAMETERS = {
    "dataset_path": os.getenv(
        "DATASET_PATH",
        os.path.join(BASE_DIR, "../../data/processed/training_dataset.json"),
    ),
    "logging_dir": os.getenv("LOGGING_DIR", os.path.join(BASE_DIR, "../../logs/")),
    "models": {
        "general_text_generation": {
            "dev": os.getenv("GENERAL_TEXT_MODEL_DEV", "gpt2-medium"),
            "prod": os.getenv("GENERAL_TEXT_MODEL_PROD", "gpt2-medium"),
        }[APP_ENV],
        "text_analysis": {
            "dev": os.getenv("TEXT_ANALYSIS_MODEL_DEV", "distilbert-base-uncased"),
            "prod": os.getenv("TEXT_ANALYSIS_MODEL_PROD", "bert-base-uncased"),
        }[APP_ENV],
        "code_generation": {
            "dev": os.getenv(
                "CODE_GENERATION_MODEL_DEV", "Salesforce/codegen-350M-multi"
            ),
            "prod": os.getenv(
                "CODE_GENERATION_MODEL_PROD", "Salesforce/codegen-2B-multi"
            ),
        }[APP_ENV],
    },
}
