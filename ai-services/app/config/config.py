import os
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

# Vérification du token Hugging Face
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError(
        "The Hugging Face token (HF_TOKEN) is missing. Check your .env file."
    )

# Logger global
DEFAULT_LOGGING_LEVEL = os.getenv("LOGGING_LEVEL", "INFO")

# Autres configurations globales essentielles (si nécessaire)
DATASET_PATH = os.getenv("DATASET_PATH", None)
if not DATASET_PATH:
    raise ValueError(
        "The DATASET_PATH environment variable is missing. Check your .env file."
    )

LOGGING_DIR = os.getenv("LOGGING_DIR", "logs/")
