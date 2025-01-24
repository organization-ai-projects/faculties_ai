from abc import ABC, abstractmethod
from typing import List, Optional


class DirectoryCleanerInterface(ABC):
    """
    Interface pour le nettoyage des répertoires.
    """

    @abstractmethod
    def clean_directory(
        self, directory: str, file_extensions: Optional[List[str]] = None
    ) -> None:
        """
        Nettoie un répertoire donné.

        :param directory: Chemin du répertoire à nettoyer.
        :param file_extensions: Liste des extensions de fichiers à supprimer.
        """
        pass
