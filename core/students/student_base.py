# core/students/student_base.py
import torch
from torch import nn

class StudentBase(nn.Module):
    """
    Classe de base pour le modèle étudiant.
    """

    def __init__(self, model: nn.Module):
        """
        Initialise un modèle étudiant.
        :param model: le modèle PyTorch à utiliser.
        """
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Effectue une passe avant avec le modèle étudiant.
        :param input_ids: Entrées (batch de séquences).
        :return: Logits du modèle.
        """
        return self.model(input_ids)
