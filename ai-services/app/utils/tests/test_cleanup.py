import os

import pytest
from app.utils.cleanup.directory_cleaner import DirectoryCleaner


@pytest.fixture
def temp_directory(tmp_path):
    """
    Crée un répertoire temporaire pour les tests.
    """
    directory = tmp_path / "test_directory"
    directory.mkdir()
    (directory / "file1.log").write_text("Log file")
    (directory / "file2.json").write_text("JSON file")
    (directory / "subdir").mkdir()
    yield directory


def test_clean_directory_removes_files(temp_directory):
    """
    Vérifie que les fichiers avec des extensions spécifiques sont supprimés.
    """
    cleaner = DirectoryCleaner()
    cleaner.clean_directory(str(temp_directory), [".log", ".json"])
    assert not any(
        file.suffix in [".log", ".json"] for file in temp_directory.rglob("*")
    )


def test_clean_directory_handles_empty_directory(tmp_path):
    """
    Vérifie que la fonction fonctionne sur un répertoire vide.
    """
    empty_directory = tmp_path / "empty_directory"
    empty_directory.mkdir()
    cleaner = DirectoryCleaner()
    cleaner.clean_directory(str(empty_directory))
    assert empty_directory.exists()


def test_clean_directory_handles_nonexistent_directory():
    """
    Vérifie que la fonction gère correctement les répertoires inexistants.
    """
    cleaner = DirectoryCleaner()
    cleaner.clean_directory("nonexistent_directory")
    assert not os.path.exists("nonexistent_directory")
