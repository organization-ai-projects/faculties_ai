import logging
import os
import shutil

from app.utils.cleaning.interfaces.dataset_resetter_interface import (
    DatasetResetterInterface,
)


class DatasetResetter(DatasetResetterInterface):
    """
    Implémentation pour la réinitialisation des datasets.
    """

    def __init__(self, logger: logging.Logger = None):
        """
        Initialise le DatasetResetter.

        :param logger: Instance de logger à utiliser.
        """
        self.logger = logger or logging.getLogger("DatasetResetter")
        self._configure_logger()

    def reset_dataset(self, original_path: str, target_path: str) -> None:
        """
        Copie un dataset depuis une source d'origine vers un chemin cible.

        :param original_path: Chemin du dataset original.
        :param target_path: Chemin où réinitialiser le dataset.
        """
        try:
            if not original_path or not target_path:
                self.logger.error("Both original_path and target_path must be defined.")
                raise ValueError("Both original_path and target_path must be defined.")

            if not os.path.exists(original_path):
                self.logger.error(f"Original dataset not found: {original_path}")
                return

            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            shutil.copy(original_path, target_path)
            self.logger.info(f"Dataset reset from {original_path} to {target_path}")

        except Exception as e:
            self.logger.error(f"Error during dataset reset: {e}")
            raise

    @staticmethod
    def _configure_logger():
        """
        Configure le logger si nécessaire.
        """
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )
