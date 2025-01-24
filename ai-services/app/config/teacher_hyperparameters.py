import os
from dotenv import load_dotenv

# Charger les variables d'environnement
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(dotenv_path=os.path.join(BASE_DIR, "../../.env"))

# Création du répertoire des enseignants
os.makedirs(
    os.getenv("TEACHER_SAVE_DIR", os.path.join(BASE_DIR, "../models/teachers/")),
    exist_ok=True,
)

TEACHER_HYPERPARAMETERS = {
    "learning_rate": float(os.getenv("TEACHER_LEARNING_RATE", 5e-5)),
    "batch_size": int(os.getenv("TEACHER_BATCH_SIZE", 4)),
    "epochs": int(os.getenv("TEACHER_EPOCHS", 3)),
    "save_dir": os.getenv(
        "TEACHER_SAVE_DIR", os.path.join(BASE_DIR, "../models/teachers/")
    ),
    "roles": {
        "basic_coding": {
            "model": os.getenv("BASIC_CODING_MODEL", "gpt2-medium"),
            "task": "text_completion",
        },
        "debugging": {
            "model": os.getenv("DEBUGGING_MODEL", "Salesforce/codegen-350M-multi"),
            "task": "code_generation",
        },
        "multilingual_tasks": {
            "model": os.getenv("MULTILINGUAL_MODEL", "facebook/m2m100_418M"),
            "task": "translation",
        },
    },
}
