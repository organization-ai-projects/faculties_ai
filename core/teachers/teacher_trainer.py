# core/teachers/teacher_trainer.py
from core.utils.base_trainer import BaseTrainer
from core.utils.logger import Logger

class TeacherTrainer(BaseTrainer):
    """
    Classe responsable de l'entraînement des professeurs IA.
    Hérite de BaseTrainer pour gérer les tâches communes d'entraînement.
    """

    def __init__(self, teacher_model, learning_rate=5e-5, batch_size=4, epochs=3):
        """
        Initialise le TeacherTrainer avec les paramètres spécifiques.
        :param teacher_model: Le modèle du professeur.
        :param learning_rate: Taux d'apprentissage.
        :param batch_size: Taille des lots.
        :param epochs: Nombre d'époques.
        """
        super().__init__(teacher_model, learning_rate)
        self.batch_size = batch_size
        self.epochs = epochs
        self.logger = Logger.get_logger(__name__)  # Utilisation du logger centralisé

    def train(self, train_dataloader):
        """
        Lance l'entraînement du professeur IA.
        :param train_dataloader: Dataloader des données d'entraînement.
        """
        self.logger.info("Début de l'entraînement des professeurs IA.")
        self.configure_scheduler(train_dataloader, self.epochs)

        for epoch in range(self.epochs):
            avg_loss = self.train_one_epoch(train_dataloader, self.compute_loss)
            self.logger.info(f"Epoch {epoch + 1}, Training Loss: {avg_loss}")

    def compute_loss(self, outputs, labels):
        """
        Calcule la perte entre les sorties du modèle et les attentes (labels).
        Surcharge la méthode de BaseTrainer si nécessaire.
        """
        loss_fn = torch.nn.CrossEntropyLoss()
        return loss_fn(outputs.view(-1, self.model.output_head.out_features), labels.view(-1))
