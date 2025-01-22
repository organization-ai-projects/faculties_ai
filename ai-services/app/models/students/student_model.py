import torch
import torch.nn as nn
from app.components.transformer_layer import TransformerLayer
from app.components.rmsnorm import RMSNorm
from app.components.rotary_position_embedding import RotaryPositionEmbedding


class StudentModel(nn.Module):
    """
    Modèle de base représentant un étudiant dans le contexte de l'IA.
    """
    def __init__(
        self,
        vocab_size: int = 50257,
        hidden_size: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        ff_multiplier: int = 4,
    ):
        super(StudentModel, self).__init__()
        self.hidden_size = hidden_size

        # Embedding des tokens
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)

        # Embedding positionnel rotary
        self.position_embedding = RotaryPositionEmbedding(max_seq_len, hidden_size)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Couches Transformer pour traiter les séquences
        self.transformer_layers = nn.ModuleList([
            TransformerLayer(hidden_size, num_heads, dropout, ff_multiplier)
            for _ in range(num_layers)
        ])

        # Normalisation finale
        self.norm = RMSNorm(hidden_size)

        # Tête de sortie pour produire les logits
        self.output_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Passe avant du modèle.
        :param input_ids: Tensor contenant les identifiants des tokens.
        :return: Logits (sortie du modèle).
        """
        x = self.token_embedding(input_ids)
        x = self.position_embedding(x)
        x = self.dropout(x)

        for layer in self.transformer_layers:
            x = layer(x)

        x = self.norm(x)
        logits = self.output_head(x)
        return logits
