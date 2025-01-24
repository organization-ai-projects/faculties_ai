from typing import Protocol


class DatasetResetterInterface(Protocol):
    """
    Interface pour la réinitialisation des datasets.
    """

    def reset_dataset(self, original_path: str, target_path: str) -> None:
        """
        Copie un dataset depuis une source d'origine vers un chemin cible.

        :param original_path: Chemin du dataset original.
        :param target_path: Chemin où réinitialiser le dataset.
        """
        pass
