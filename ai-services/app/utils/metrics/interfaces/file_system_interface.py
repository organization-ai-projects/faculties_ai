from typing import Protocol


class FileSystemInterface(Protocol):
    """
    Interface pour les opérations sur le système de fichiers.
    """

    def exists(self, path: str) -> bool:
        pass

    def makedirs(self, path: str, exist_ok: bool = True) -> None:
        pass

    def join(self, *paths: str) -> str:
        pass
