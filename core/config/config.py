import os
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

# Vérification du token Hugging Face
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("Le token Hugging Face (HF_TOKEN) est manquant. Vérifiez votre fichier .env.")
else:
    print("Token Hugging Face chargé avec succès!")

# Hyperparamètres centralisés
HYPERPARAMETERS = {
    "models": {
        "general_text_generation": ["gpt2-medium", "EleutherAI/gpt-neo-125M"],
        "text_analysis": ["bert-base-uncased", "distilbert-base-uncased"],
        "multilingual_tasks": ["facebook/m2m100_418M"],
        "code_generation": ["Salesforce/codegen-350M-multi"]
    },
    "learning_rate": 5e-5,
    "batch_size": 4,
    "epochs": 3,
    "dataset_path": "data/processed/training_dataset.json",
    "output_dir": "models/",
    "logging_dir": "logs/"
}
