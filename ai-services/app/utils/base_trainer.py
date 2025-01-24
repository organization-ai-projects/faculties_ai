# core/utils/base_trainer.py
import torch
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from core.utils.logger import Logger


class BaseTrainer:
    """
    Classe de base pour gérer l'entraînement générique d'un modèle.
    """

    def __init__(self, model, learning_rate=5e-5):
        """
        Initialise le modèle, l'optimiseur et le scheduler.
        :param model: Modèle à entraîner.
        :param learning_rate: Taux d'apprentissage.
        """
        self.model = model
        self.learning_rate = learning_rate
        self.optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = None
        self.logger = Logger.get_logger(self.__class__.__name__)

    def configure_scheduler(self, train_dataloader, epochs):
        """
        Configure le scheduler pour l'entraînement.
        :param train_dataloader: Dataloader des données d'entraînement.
        :param epochs: Nombre d'époques.
        """
        total_steps = len(train_dataloader) * epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=0, num_training_steps=total_steps
        )
        self.logger.info(f"[Scheduler] Configuré pour {total_steps} étapes.")

    def train_one_epoch(self, dataloader, compute_loss_fn):
        """
        Entraîne le modèle pour une époque donnée.
        :param dataloader: Dataloader des données d'entraînement.
        :param compute_loss_fn: Fonction pour calculer la perte.
        """
        self.model.train()
        total_loss = 0

        self.logger.info("[Époque] Début de l'entraînement.")
        for batch in dataloader:
            input_ids = batch["input_ids"]
            labels = batch["labels"]

            # Forward pass
            outputs = self.model(input_ids)
            loss = compute_loss_fn(outputs, labels)

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        self.logger.info(f"[Époque] Terminée. Perte moyenne : {avg_loss:.4f}")
        return avg_loss

    def compute_loss(self, outputs, labels):
        """
        Calcule la perte entre les prédictions et les labels.
        Peut être surchargée par les classes dérivées.
        """
        loss_fn = torch.nn.CrossEntropyLoss()
        return loss_fn(
            outputs.view(-1, self.model.output_head.out_features), labels.view(-1)
        )
