import os

# import shutil
import pytest
from scripts.dataset_reset import reset_dataset


@pytest.fixture
def setup_directories(tmp_path):
    """
    Prépare les répertoires de test et les fichiers pour les tests.
    """
    # Répertoire temporaire pour les tests
    original_dir = tmp_path / "original"
    target_dir = tmp_path / "target"
    original_dir.mkdir()
    target_dir.mkdir()

    # Créer un fichier dataset original
    original_file = original_dir / "training_dataset.json"
    with open(original_file, "w") as f:
        f.write('{"data": "original content"}')

    # Retourne les chemins pour les tests
    return str(original_file), str(target_dir / "training_dataset.json")


def test_reset_dataset_success(setup_directories):
    """
    Teste la réinitialisation réussie d'un dataset.
    """
    original_path, target_path = setup_directories

    # Appelle la fonction à tester
    reset_dataset(original_path, target_path)

    # Vérifie que le fichier a bien été copié
    assert os.path.exists(target_path), "Le fichier n'a pas été copié."
    with open(target_path, "r") as f:
        content = f.read()
    assert (
        content == '{"data": "original content"}'
    ), "Le contenu du fichier est incorrect."


def test_reset_dataset_missing_original(setup_directories):
    """
    Teste le comportement de la fonction lorsque le dataset original est manquant.
    """
    original_path, target_path = setup_directories

    # Supprime le fichier original pour simuler l'absence
    os.remove(original_path)

    # Appelle la fonction et capture la sortie
    reset_dataset(original_path, target_path)

    # Vérifie que le fichier cible n'existe pas
    assert not os.path.exists(
        target_path
    ), "Le fichier cible a été créé alors qu'il ne devait pas l'être."
