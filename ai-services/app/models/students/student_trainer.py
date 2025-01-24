from app.utils.base_trainer import BaseTrainer
from app.utils.logger import Logger
from app.utils.metrics_handler import MetricsHandler
from app.utils.evaluator import Evaluator
from app.models.teachers.professor_collaborator import ProfessorCollaborator
from app.config.hyperparameters import HYPERPARAMETERS
from torch.utils.data import DataLoader
import torch
import os


class StudentTrainer(BaseTrainer):
    """
    Handles training for AI student models with modular components.
    """

    def __init__(
        self,
        student_model: torch.nn.Module,
        train_dataset: torch.utils.data.Dataset,
        val_dataset: torch.utils.data.Dataset,
        professors: list[torch.nn.Module] = None,
    ):
        """
        Initializes the StudentTrainer with modular components.
        """
        params = HYPERPARAMETERS["students"]
        super().__init__(student_model, params["learning_rate"])

        if not train_dataset or len(train_dataset) == 0:
            raise ValueError("The training dataset must not be empty.")
        if not val_dataset or len(val_dataset) == 0:
            raise ValueError("The validation dataset must not be empty.")

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.professors = professors or []
        self.epochs = params["epochs"]
        self.batch_size = params["batch_size"]
        self.save_dir = params["save_dir"]
        self.logger = Logger.get_logger(__name__)
        self.metrics_handler = MetricsHandler(self.save_dir)
        self.professor_collaborator = ProfessorCollaborator()
        self.evaluator = Evaluator(self.model)
        self.best_loss = float("inf")
        os.makedirs(self.save_dir, exist_ok=True)

    def train(self):
        """
        Starts the training process.
        """
        self.logger.info("Starting the training process.")
        train_dataloader = DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True
        )
        val_dataloader = DataLoader(self.val_dataset, batch_size=self.batch_size)

        for epoch in range(self.epochs):
            self.logger.info(f"Epoch {epoch + 1}/{self.epochs}")
            avg_loss = self.train_one_epoch(
                train_dataloader, self.evaluator.compute_loss
            )
            self.logger.info(f"Training Loss: {avg_loss:.4f}")
            val_loss, metrics = self.evaluator.evaluate(val_dataloader)

            self.metrics_handler.save_metrics(
                {"epoch": epoch + 1, "val_loss": val_loss, **metrics}
            )

            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.save_model(epoch, is_best=True)

            for professor in self.professors:
                self.professor_collaborator.collaborate(self.model, professor)

    def save_model(self, epoch, is_best=False):
        """
        Saves the model.
        """
        model_filename = (
            "student_model_best.pth"
            if is_best
            else f"student_model_epoch_{epoch + 1}.pth"
        )
        model_path = os.path.join(self.save_dir, model_filename)
        torch.save(self.model.state_dict(), model_path)
        self.logger.info(f"Model saved at: {model_path}")
