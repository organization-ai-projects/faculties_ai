import pytest
from scripts.dataset_reset import reset_dataset


@pytest.fixture
def dataset_paths(tmp_path):
    """
    Crée un chemin pour les datasets originaux et cibles.
    """
    original_dataset = tmp_path / "original_dataset.json"
    target_dataset = tmp_path / "target_dataset.json"
    original_dataset.write_text("Dummy dataset content")
    return original_dataset, target_dataset


def test_reset_dataset(dataset_paths):
    """
    Vérifie que le dataset est bien copié du chemin source vers le chemin cible.
    """
    original, target = dataset_paths
    reset_dataset(str(original), str(target))
    assert target.exists()
    assert target.read_text() == "Dummy dataset content"


def test_reset_dataset_missing_source(dataset_paths):
    """
    Vérifie qu'une erreur est loguée si la source n'existe pas.
    """
    _, target = dataset_paths
    with pytest.raises(Exception):
        reset_dataset("nonexistent_dataset.json", str(target))
