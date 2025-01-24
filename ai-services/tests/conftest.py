# Répertoires et chemins utilisés pour les tests
from dotenv import load_dotenv
import os

import pytest
from scripts.directory_cleaner import clean_directory
from scripts.dataset_reset import reset_dataset

# Extensions de fichiers à nettoyer
FILE_EXTENSIONS_TO_REMOVE = [".pth", ".log", ".json"]


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(dotenv_path=os.path.join(BASE_DIR, "../.env.test"))

LOGGING_DIR = os.getenv("LOGGING_DIR")
TEACHER_SAVE_DIR = os.getenv("TEACHER_SAVE_DIR")
STUDENT_SAVE_DIR = os.getenv("STUDENT_SAVE_DIR")
DATASET_PATH = os.getenv("DATASET_PATH")
ORIGINAL_DATASET_PATH = os.path.join(BASE_DIR, "../data/original/training_dataset.json")

# Répertoires à nettoyer
TEST_DIRECTORIES = [LOGGING_DIR, TEACHER_SAVE_DIR, STUDENT_SAVE_DIR]


@pytest.fixture(scope="session", autouse=True)
def setup_and_teardown():
    """
    Nettoie l'environnement et réinitialise les datasets avant et après les tests.
    """
    print("\n[TEST SETUP] Cleaning test environment and resetting datasets...")

    # Nettoyage des répertoires
    for directory in TEST_DIRECTORIES:
        clean_directory(directory, file_extensions=FILE_EXTENSIONS_TO_REMOVE)

    # Réinitialisation des datasets
    reset_dataset(ORIGINAL_DATASET_PATH, DATASET_PATH)

    yield  # Code après cette ligne s'exécute une fois tous les tests terminés

    print("\n[TEST TEARDOWN] Cleaning test environment...")
    for directory in TEST_DIRECTORIES:
        clean_directory(directory, file_extensions=FILE_EXTENSIONS_TO_REMOVE)
