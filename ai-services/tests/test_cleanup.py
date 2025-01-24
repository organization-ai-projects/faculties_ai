import os
import shutil
from dotenv import load_dotenv

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


def clean_test_environment():
    """
    Supprime les fichiers générés pendant les tests et réinitialise les répertoires.
    """
    for directory in TEST_DIRECTORIES:
        if directory and os.path.exists(directory):
            # Supprimer les fichiers avec des extensions spécifiques
            for root, _, files in os.walk(directory):
                for file in files:
                    if file.endswith(tuple(FILE_EXTENSIONS_TO_REMOVE)):
                        os.remove(os.path.join(root, file))
                        print(f"Deleted file: {os.path.join(root, file)}")

            # Réinitialiser le répertoire
            shutil.rmtree(directory)
            os.makedirs(directory, exist_ok=True)
            print(f"Cleaned and recreated directory: {directory}")
        else:
            print(f"Directory does not exist or is not defined: {directory}")


def reset_test_datasets():
    """
    Réinitialise les datasets de test à partir des fichiers originaux.
    """
    original_dataset_path = os.path.join(
        BASE_DIR, "../data/original/training_dataset.json"
    )
    if os.path.exists(original_dataset_path) and DATASET_PATH:
        shutil.copy(original_dataset_path, DATASET_PATH)
        print(f"Dataset reset from {original_dataset_path} to {DATASET_PATH}")
    else:
        print("Original dataset not found or target dataset path not defined.")


if __name__ == "__main__":
    print("Cleaning test environment...")
    clean_test_environment()
    print("Resetting test datasets...")
    reset_test_datasets()
