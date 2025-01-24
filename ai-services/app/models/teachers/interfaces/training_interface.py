from abc import ABC, abstractmethod


class TrainingInterface(ABC):
    """
    Interface pour définir les méthodes nécessaires à tout service d'entraînement.
    """

    @abstractmethod
    def train(self, model_name: str, epochs: int, learning_rate: float) -> str:
        """
        Lance l'entraînement d'un modèle IA.
        :param model_name: Nom du modèle à entraîner.
        :param epochs: Nombre d'époques d'entraînement.
        :param learning_rate: Taux d'apprentissage utilisé.
        :return: Résultat sous forme de message ou d'objet.
        """
        pass
