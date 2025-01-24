import logging
import os
import shutil
import stat
from typing import List, Optional

from app.utils.cleaning.interfaces.cleaner_interface import DirectoryCleanerInterface


class DirectoryCleaner(DirectoryCleanerInterface):
    """
    Implémentation pour le nettoyage des répertoires.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialise le DirectoryCleaner.

        :param logger: Instance de logger à utiliser. Si aucun logger n'est fourni,
                       un logger par défaut sera utilisé.
        """
        self.logger = logger or self._get_default_logger()

    def clean_directory(
        self, directory: str, file_extensions: Optional[List[str]] = None
    ) -> None:
        """
        Supprime les fichiers spécifiques et réinitialise un répertoire.

        :param directory: Chemin du répertoire à nettoyer.
        :param file_extensions: Liste des extensions de fichiers à supprimer.
        """
        if not os.path.exists(directory):
            self.logger.warning(
                f"Directory does not exist or is not defined: {directory}"
            )
            return

        try:
            files_deleted = self._delete_files(directory, file_extensions)
            self._remove_empty_directories(directory)

            if files_deleted and not os.listdir(directory):
                shutil.rmtree(directory)
                os.makedirs(directory, exist_ok=True)
                self.logger.info(f"Cleaned and recreated directory: {directory}")
            elif not files_deleted:
                self.logger.info(
                    f"No files matching the extensions were found in {directory}"
                )

        except Exception as e:
            self.logger.error(
                f"An error occurred while cleaning directory {directory}: {e}"
            )
            raise

    def _delete_files(
        self, directory: str, file_extensions: Optional[List[str]]
    ) -> bool:
        """
        Supprime les fichiers correspondant aux extensions spécifiées dans un répertoire.

        :param directory: Chemin du répertoire à nettoyer.
        :param file_extensions: Liste des extensions de fichiers à supprimer.
        :return: True si des fichiers ont été supprimés, False sinon.
        """
        files_deleted = False

        for root, _, files in os.walk(directory, topdown=False):
            for file in files:
                file_path = os.path.join(root, file)
                if file_extensions is None or any(
                    file.endswith(ext) for ext in file_extensions
                ):
                    if self._delete_file(file_path):
                        files_deleted = True

        return files_deleted

    def _delete_file(self, file_path: str) -> bool:
        """
        Supprime un fichier spécifique.

        :param file_path: Chemin du fichier à supprimer.
        :return: True si le fichier a été supprimé, False sinon.
        """
        try:
            os.chmod(file_path, stat.S_IWRITE)
            os.remove(file_path)
            self.logger.info(f"Deleted file: {file_path}")
            return True
        except OSError as e:
            self.logger.error(f"Error deleting file {file_path}: {e}")
            return False

    def _remove_empty_directories(self, directory: str) -> None:
        """
        Supprime les sous-répertoires vides dans un répertoire donné.

        :param directory: Chemin du répertoire à nettoyer.
        """
        for root, dirs, _ in os.walk(directory, topdown=False):
            for dir_ in dirs:
                dir_path = os.path.join(root, dir_)
                if not os.listdir(dir_path):
                    self._delete_directory(dir_path)

    def _delete_directory(self, dir_path: str) -> None:
        """
        Supprime un répertoire vide.

        :param dir_path: Chemin du répertoire à supprimer.
        """
        try:
            os.rmdir(dir_path)
            self.logger.info(f"Deleted empty directory: {dir_path}")
        except OSError as e:
            self.logger.error(f"Error removing directory {dir_path}: {e}")

    @staticmethod
    def _get_default_logger() -> logging.Logger:
        """
        Crée et retourne un logger par défaut.

        :return: Instance de logger par défaut.
        """
        logger = logging.getLogger("DirectoryCleaner")
        if not logger.hasHandlers():
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
