from app.utils.base_trainer import BaseTrainer
from app.utils.logger import Logger
from app.config.hyperparameters import HYPERPARAMETERS
from torch.utils.data import DataLoader
import torch
import os


class StudentTrainer(BaseTrainer):
    """
    Handles training for AI student models.
    """

    def __init__(self, student_model: torch.nn.Module, train_dataset: torch.utils.data.Dataset, val_dataset: torch.utils.data.Dataset):
        """
        Initializes the StudentTrainer.
        """
        params = HYPERPARAMETERS['students']
        super().__init__(student_model, params['learning_rate'])

        # Vérification des datasets
        if not train_dataset or len(train_dataset) == 0:
            raise ValueError("The training dataset must not be empty.")
        if not val_dataset or len(val_dataset) == 0:
            raise ValueError("The validation dataset must not be empty.")

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.epochs = params['epochs']
        self.batch_size = params['batch_size']
        self.save_dir = params['save_dir']
        self.logger = Logger.get_logger(__name__)

        os.makedirs(self.save_dir, exist_ok=True)

    def train(self):
        """
        Starts the training process.
        """
        self.logger.info("Starting the training process.")
        train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        val_dataloader = DataLoader(self.val_dataset, batch_size=self.batch_size)

        for epoch in range(self.epochs):
            self.logger.info(f"Epoch {epoch + 1}/{self.epochs}")
            avg_loss = self.train_one_epoch(train_dataloader, self.compute_loss)
            self.logger.info(f"Training Loss: {avg_loss:.4f}")
            self.evaluate(val_dataloader, epoch)
            self.save_model(epoch)

    def evaluate(self, dataloader, epoch):
        """
        Evaluates the model.
        """
        self.logger.info(f"Evaluating epoch {epoch + 1}")
        total_loss = 0.0
        self.model.eval()
        with torch.no_grad():
            for batch in dataloader:
                if "input_ids" not in batch or "labels" not in batch:
                    self.logger.warning("Batch is missing 'input_ids' or 'labels'. Skipping.")
                    continue
                inputs, labels = batch["input_ids"], batch["labels"]
                outputs = self.model(inputs)
                loss = self.compute_loss(outputs, labels)
                total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        self.logger.info(f"Validation Loss: {avg_loss:.4f}")

    def save_model(self, epoch):
        """
        Saves the model.
        """
        model_path = os.path.join(self.save_dir, f"student_model_epoch_{epoch + 1}.pth")
        torch.save(self.model.state_dict(), model_path)
        self.logger.info(f"Model saved at: {model_path}")
