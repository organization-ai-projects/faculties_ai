import os

from app.utils.metrics.interfaces.file_system_interface import FileSystemInterface


class OSFileSystem(FileSystemInterface):
    """
    Implémentation par défaut basée sur le module os pour le système de fichiers.
    """

    def exists(self, path: str) -> bool:
        return os.path.exists(path)

    def makedirs(self, path: str, exist_ok: bool = True) -> None:
        os.makedirs(path, exist_ok=exist_ok)

    def join(self, *paths: str) -> str:
        return os.path.join(*paths)
