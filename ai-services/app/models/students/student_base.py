import torch
from torch import nn
from torch.nn import Module
from app.utils.logger import Logger


class StudentBase(nn.Module):
    """
    Base class for the student model.
    """

    def __init__(self, model: Module):
        """
        Initializes a student model.
        :param model: the PyTorch model to use.
        """
        super().__init__()
        if not isinstance(model, nn.Module):
            raise TypeError("model must be an instance of torch.nn.Module")
        self.model = model

        # Utilisation du logger global
        self.logger = Logger.setup_logger(__name__)
        self.logger.info("StudentBase model initialized successfully.")

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass with the student model.
        :param input_ids: Inputs (batch of sequences).
        :return: Model logits.
        """
        if not isinstance(input_ids, torch.Tensor):
            raise TypeError("input_ids must be a torch.Tensor")

        try:
            logits = self.model(input_ids)
            return logits
        except Exception as e:
            self.logger.error(f"Error during forward pass: {e}")
            raise
