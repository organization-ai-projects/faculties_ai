import os
from dotenv import load_dotenv
from scripts.directory_cleaner import clean_directory
from scripts.dataset_reset import reset_dataset

# Charger les variables d'environnement
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(dotenv_path=os.path.join(BASE_DIR, "../.env.test"))

# Chemins définis dans le .env.test
DATASET_PATH = os.getenv("DATASET_PATH")
LOGGING_DIR = os.getenv("LOGGING_DIR")
TEACHER_SAVE_DIR = os.getenv("TEACHER_SAVE_DIR")
STUDENT_SAVE_DIR = os.getenv("STUDENT_SAVE_DIR")

# Extensions de fichiers à nettoyer (ajustable)
FILE_EXTENSIONS_TO_REMOVE = [".pth", ".log", ".json"]

# Répertoires à nettoyer
TEST_DIRECTORIES = [LOGGING_DIR, TEACHER_SAVE_DIR, STUDENT_SAVE_DIR]

if __name__ == "__main__":
    # Nettoyage des répertoires
    print("Cleaning test environment...")
    for directory in TEST_DIRECTORIES:
        clean_directory(directory, file_extensions=FILE_EXTENSIONS_TO_REMOVE)

    # Réinitialisation des datasets
    print("Resetting test datasets...")
    reset_dataset(
        original_path=os.path.join(BASE_DIR, "../data/original/training_dataset.json"),
        target_path=DATASET_PATH,
    )
