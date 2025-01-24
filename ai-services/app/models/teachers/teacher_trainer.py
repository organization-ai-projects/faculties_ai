from app.utils.base_trainer import BaseTrainer
from app.utils.logger import Logger
import torch
from app.config.hyperparameters import HYPERPARAMETERS


class TeacherTrainer(BaseTrainer):
    """
    Class responsible for training AI teachers.
    Inherits from BaseTrainer to handle common training tasks.
    """

    def __init__(self, teacher_model: torch.nn.Module):
        """
        Initializes the TeacherTrainer with specific parameters.
        :param teacher_model: The teacher model.
        """
        params = HYPERPARAMETERS["teachers"]
        super().__init__(teacher_model, params["learning_rate"])
        self.batch_size = params["batch_size"]
        self.epochs = params["epochs"]
        self.save_dir = params["save_dir"]
        self.logger = Logger.setup_logger(__name__)  # Centralized logger

    def train(self, train_dataloader: torch.utils.data.DataLoader):
        """
        Starts training the AI teacher.
        :param train_dataloader: Training data dataloader.
        """
        try:
            self.logger.info("Starting AI teacher training.")
            self.configure_scheduler(train_dataloader, self.epochs)

            for epoch in range(self.epochs):
                self.logger.info(f"Starting epoch {epoch + 1}/{self.epochs}.")
                avg_loss = self.train_one_epoch(train_dataloader, self.compute_loss)
                self.logger.info(f"Epoch {epoch + 1}, Training Loss: {avg_loss}")
                self.save_model(epoch)

            self.logger.info("Training successfully completed.")
        except Exception as e:
            self.logger.error(f"Error during training: {e}")
            raise

    def compute_loss(self, outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Computes the loss between model outputs and expectations (labels).
        Overrides the method from BaseTrainer if necessary.
        """
        try:
            loss_fn = torch.nn.CrossEntropyLoss()
            return loss_fn(
                outputs.view(-1, self.model.output_head.out_features), labels.view(-1)
            )
        except Exception as e:
            self.logger.error(f"Error during loss computation: {e}")
            raise

    def save_model(self, epoch: int):
        """
        Saves the teacher model after each epoch.
        """
        try:
            model_path = f"{self.save_dir}/teacher_epoch_{epoch + 1}.pth"
            torch.save(self.model.state_dict(), model_path)
            self.logger.info(f"Model saved: {model_path}")
        except Exception as e:
            self.logger.error(f"Error during model saving: {e}")
            raise
