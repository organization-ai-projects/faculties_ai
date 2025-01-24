# Chemin : app/utils/metrics/interfaces/metrics_storage_interface.py

from typing import Any, Dict


class MetricsStorageInterface:
    """
    Interface pour le stockage des métriques.
    """

    def save(self, data: Dict[str, Any], path: str) -> None:
        """
        Sauvegarde des données dans un fichier.

        :param data: Données à sauvegarder.
        :param path: Chemin du fichier.
        """
        raise NotImplementedError

    def load(self, path: str) -> Dict[str, Any]:
        """
        Chargement des données depuis un fichier.

        :param path: Chemin du fichier.
        :return: Données chargées sous forme de dictionnaire.
        """
        raise NotImplementedError
