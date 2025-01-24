from abc import ABC, abstractmethod


class InferenceInterface(ABC):
    """
    Interface pour tous les services d'inférence (local ou distant).
    """

    @abstractmethod
    def predict(self, text: str) -> str:
        """
        Méthode pour effectuer une prédiction sur une chaîne de texte donnée.
        :param text: Texte d'entrée pour l'inférence.
        :return: Résultat de la prédiction sous forme de chaîne.
        """
        pass
