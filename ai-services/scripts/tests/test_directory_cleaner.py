import os

import pytest
from scripts.directory_cleaner import clean_directory


@pytest.fixture
def test_directory(tmp_path):
    """
    Crée un répertoire temporaire pour les tests.
    """
    directory = tmp_path / "test_directory"
    directory.mkdir()
    # Créer des fichiers temporaires avec différentes extensions
    (directory / "file1.txt").write_text("Test file 1")
    (directory / "file2.log").write_text("Test file 2")
    (directory / "file3.json").write_text("Test file 3")
    (directory / "file4.txt").write_text("Test file 4")
    yield directory
    # Nettoyage après les tests (pytest s'en charge automatiquement)


def test_clean_directory_removes_files(test_directory):
    """
    Vérifie que les fichiers avec des extensions spécifiées sont supprimés.
    """
    file_extensions = [".txt", ".log"]
    clean_directory(test_directory, file_extensions)

    # Vérifie que seuls les fichiers spécifiés sont supprimés
    remaining_files = list(test_directory.iterdir())
    assert len(remaining_files) == 1
    assert remaining_files[0].name == "file3.json"


def test_clean_directory_recreates_directory(test_directory):
    """
    Vérifie que le répertoire est recréé après le nettoyage.
    """
    file_extensions = [".txt", ".log", ".json"]
    clean_directory(test_directory, file_extensions)

    # Vérifie que le répertoire a été recréé
    assert test_directory.exists()
    assert len(list(test_directory.iterdir())) == 0


def test_clean_directory_handles_nonexistent_directory():
    """
    Vérifie que la fonction gère correctement les répertoires inexistants.
    """
    nonexistent_directory = "nonexistent_dir"
    clean_directory(nonexistent_directory, [".txt"])  # Ne doit pas lever d'erreur
    assert not os.path.exists(nonexistent_directory)


def test_clean_directory_empty_directory(tmp_path):
    """
    Vérifie que la fonction fonctionne sur un répertoire vide.
    """
    empty_directory = tmp_path / "empty_directory"
    empty_directory.mkdir()
    clean_directory(empty_directory, [".txt"])
    assert empty_directory.exists()


def test_clean_directory_files_without_extensions(test_directory):
    """
    Vérifie que les fichiers sans extension ne sont pas affectés.
    """
    (test_directory / "file_no_ext").write_text("No extension")
    clean_directory(test_directory, [".txt"])
    assert (test_directory / "file_no_ext").exists()


def test_clean_directory_handles_permissions(tmp_path):
    """
    Vérifie que la fonction gère correctement les permissions restreintes.
    """
    restricted_directory = tmp_path / "restricted_directory"
    restricted_directory.mkdir()
    restricted_file = restricted_directory / "restricted_file.txt"
    restricted_file.write_text("Restricted file")
    os.chmod(restricted_file, 0o400)  # Lecture seule

    try:
        clean_directory(restricted_directory, [".txt"])
        assert not restricted_file.exists()
    except PermissionError:
        pytest.fail("La fonction n'a pas géré les permissions correctement.")


def test_clean_directory_logs_actions(test_directory, caplog):
    """
    Vérifie que les actions sont loguées correctement.
    """
    file_extensions = [".txt"]
    with caplog.at_level("INFO"):
        clean_directory(test_directory, file_extensions)
    assert "Deleted file" in caplog.text
    assert "Cleaned and recreated directory" in caplog.text


def test_clean_directory_handles_all_files(test_directory):
    """
    Vérifie que tous les fichiers sont supprimés si aucune extension n'est spécifiée.
    """
    clean_directory(test_directory)  # Pas d'extensions spécifiées
    assert len(list(test_directory.iterdir())) == 0


def test_clean_directory_handles_nested_directories(tmp_path):
    """
    Vérifie que la fonction nettoie correctement les sous-répertoires.
    """
    nested_directory = tmp_path / "nested_directory"
    nested_directory.mkdir()
    sub_dir = nested_directory / "sub_dir"
    sub_dir.mkdir()
    (sub_dir / "nested_file.txt").write_text("Nested file")

    clean_directory(nested_directory, [".txt"])
    assert len(list(nested_directory.iterdir())) == 0


def test_clean_directory_handles_rmdir_error(tmp_path):
    """Test la gestion des erreurs lors de la suppression de répertoires."""
    test_dir = tmp_path / "test_dir"
    test_dir.mkdir()
    sub_dir = test_dir / "sub_dir"
    sub_dir.mkdir()
    (sub_dir / "file.txt").write_text("Test")

    clean_directory(test_dir, [".txt"])
    assert sub_dir.exists()


def test_clean_directory_handles_general_error(tmp_path, monkeypatch):
    """Test la gestion des erreurs générales."""

    def mock_walk(*args, **kwargs):
        raise Exception("Test error")

    monkeypatch.setattr(os, "walk", mock_walk)
    test_dir = tmp_path / "test_dir"
    test_dir.mkdir()

    with pytest.raises(Exception):
        clean_directory(test_dir)
