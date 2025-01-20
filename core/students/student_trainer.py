# core/students/student_trainer.py

from core.utils.base_trainer import BaseTrainer
from core.utils.logger import Logger
from torch.utils.data import DataLoader

class StudentTrainer(BaseTrainer):
    """
    Classe responsable de l'entraînement des étudiants IA.
    Hérite de BaseTrainer pour gérer les tâches communes d'entraînement.
    """

    def __init__(self, student_model, train_dataset, val_dataset, learning_rate=5e-5, epochs=3, batch_size=4, save_dir="models/"):
        """
        Initialise le StudentTrainer avec les paramètres nécessaires.
        :param student_model: Modèle étudiant à entraîner.
        :param train_dataset: Dataset pour l'entraînement.
        :param val_dataset: Dataset pour la validation.
        :param learning_rate: Taux d'apprentissage.
        :param epochs: Nombre d'époques d'entraînement.
        :param batch_size: Taille des lots.
        :param save_dir: Répertoire pour sauvegarder les modèles.
        """
        super().__init__(student_model, learning_rate)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.epochs = epochs
        self.batch_size = batch_size
        self.save_dir = save_dir
        self.logger = Logger.get_logger(__name__)  # Utilisation du logger centralisé

    def train(self):
        """
        Lance l'entraînement des étudiants IA.
        """
        try:
            self.logger.info("Début de l'entraînement des étudiants IA.")
            train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
            val_dataloader = DataLoader(self.val_dataset, batch_size=self.batch_size)
            self.configure_scheduler(train_dataloader, self.epochs)

            for epoch in range(self.epochs):
                self.logger.info(f"Début de l'époque {epoch + 1}/{self.epochs}.")
                avg_loss = self.train_one_epoch(train_dataloader, self.compute_loss)
                self.logger.info(f"Epoch {epoch + 1}, Training Loss: {avg_loss}")
                self.evaluate(val_dataloader, epoch)
                self.save_model(epoch)

            self.logger.info("Entraînement terminé avec succès.")
        except Exception as e:
            self.logger.error(f"Erreur pendant l'entraînement : {e}")
            raise

    def evaluate(self, dataloader, epoch):
        """
        Évalue le modèle étudiant sur les données de validation.
        """
        try:
            self.logger.info(f"Début de l'évaluation pour l'époque {epoch + 1}.")
            self.model.eval()
            total_loss = 0

            with torch.no_grad():
                for batch in dataloader:
                    input_ids = batch['input_ids']
                    labels = batch['labels']

                    outputs = self.model(input_ids)
                    loss = self.compute_loss(outputs, labels)
                    total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            self.logger.info(f"Validation Loss (Epoch {epoch + 1}): {avg_loss}")
        except Exception as e:
            self.logger.error(f"Erreur pendant l'évaluation : {e}")
            raise

    def save_model(self, epoch):
        """
        Sauvegarde le modèle étudiant après chaque époque.
        """
        try:
            model_path = f"{self.save_dir}/student_epoch_{epoch + 1}.pth"
            torch.save(self.model.state_dict(), model_path)
            self.logger.info(f"Modèle sauvegardé : {model_path}")
        except Exception as e:
            self.logger.error(f"Erreur lors de la sauvegarde du modèle : {e}")
            raise
