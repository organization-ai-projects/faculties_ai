from typing import Any, Dict

from app.utils.logger import Logger
from app.utils.metrics.interfaces.metrics_storage_interface import (
    MetricsStorageInterface,
)
from app.utils.metrics.json_metrics_storage import JsonMetricsStorage
from app.utils.metrics.path_manager import PathManager


class MetricsHandler:
    """
    Gère les métriques et délègue le stockage et la gestion des chemins.
    """

    def __init__(
        self,
        path_manager: PathManager,
        storage: MetricsStorageInterface = None,
    ):
        """
        Initialise le gestionnaire de métriques.

        :param path_manager: Gestionnaire de chemins.
        :param storage: Implémentation de stockage des métriques.
        """
        self.path_manager = path_manager
        self.logger = Logger.get_logger(__name__)
        self.storage = storage or JsonMetricsStorage()

    def save_metrics(
        self, metrics: Dict[str, Any], filename: str = "metrics_history.json"
    ) -> None:
        """
        Sauvegarde les métriques dans un fichier.

        :param metrics: Données de métriques à sauvegarder.
        :param filename: Nom du fichier où sauvegarder les métriques.
        """
        try:
            self.path_manager.ensure_directory_exists()
            metrics_path = self.path_manager.get_metrics_path(filename)
            self.storage.save(metrics, metrics_path)
            self.logger.info(f"Metrics saved at: {metrics_path}")
        except Exception as e:
            self.logger.error(f"Failed to save metrics: {e}")
            raise

    def load_metrics(self, filename: str = "metrics_history.json") -> Dict[str, Any]:
        """
        Charge les métriques depuis un fichier.

        :param filename: Nom du fichier où les métriques sont sauvegardées.
        :return: Dictionnaire contenant les métriques.
        """
        try:
            metrics_path = self.path_manager.get_metrics_path(filename)
            metrics = self.storage.load(metrics_path)
            if metrics:
                self.logger.info(f"Metrics loaded from: {metrics_path}")
            else:
                self.logger.warning(f"No metrics found at: {metrics_path}")
            return metrics
        except Exception as e:
            self.logger.error(f"Failed to load metrics: {e}")
            return {}
