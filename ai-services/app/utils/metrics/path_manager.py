from app.utils.metrics.interfaces.file_system_interface import FileSystemInterface


class PathManager:
    """
    Gestionnaire des chemins pour les métriques.
    """

    def __init__(self, base_dir: str, file_system: FileSystemInterface):
        """
        Initialise le gestionnaire de chemins.

        :param base_dir: Répertoire de base.
        :param file_system: Interface pour les opérations sur le système de fichiers.
        """
        self.base_dir = base_dir
        self.file_system = file_system

    def get_metrics_path(self, filename: str) -> str:
        """
        Retourne le chemin complet pour un fichier de métriques.

        :param filename: Nom du fichier.
        :return: Chemin complet du fichier.
        """
        return self.file_system.join(self.base_dir, filename)

    def ensure_directory_exists(self) -> None:
        """
        S'assure que le répertoire de base existe.
        """
        self.file_system.makedirs(self.base_dir, exist_ok=True)
