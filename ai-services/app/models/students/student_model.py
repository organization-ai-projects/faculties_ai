import torch
import torch.nn as nn
from app.components.rmsnorm import RMSNorm
from app.components.rotary_position_embedding import RotaryPositionEmbedding
from app.components.transformer_layer import TransformerLayer
from app.utils.logger import Logger


class StudentModel(nn.Module):
    """
    Modèle de base représentant un étudiant dans le contexte de l'IA.
    """

    def __init__(
        self,
        config: dict,
        use_pretrained: bool = False,
        pretrained_model_name: str = None,
        logger: Logger = None,
    ):
        """
        Initialisation du modèle étudiant.
        :param config: Dictionnaire contenant les hyperparamètres du modèle.
        :param use_pretrained: Indique si un modèle préentraîné doit être utilisé.
        :param pretrained_model_name: Nom du modèle préentraîné.
        :param logger: Instance de logger pour les journaux.
        """
        super(StudentModel, self).__init__()
        self.hidden_size = config.get("hidden_size", 768)
        self.logger = logger or Logger.get_logger(__name__)

        # Embedding des tokens
        vocab_size = config.get("vocab_size", 50257)
        self.token_embedding = nn.Embedding(vocab_size, self.hidden_size)

        # Embedding positionnel rotary
        max_seq_len = config.get("max_seq_len", 512)
        self.position_embedding = RotaryPositionEmbedding(max_seq_len, self.hidden_size)

        # Dropout
        dropout = config.get("dropout", 0.1)
        self.dropout = nn.Dropout(dropout)

        # Initialisation des couches Transformer
        num_layers = config.get("num_layers", 12)
        num_heads = config.get("num_heads", 12)
        ff_multiplier = config.get("ff_multiplier", 4)
        self.transformer_layers = nn.ModuleList(
            [
                TransformerLayer(self.hidden_size, num_heads, dropout, ff_multiplier)
                for _ in range(num_layers)
            ]
        )

        # Normalisation finale
        self.norm = RMSNorm(self.hidden_size)

        # Tête de sortie pour produire les logits
        self.output_head = nn.Linear(self.hidden_size, vocab_size)

        # Chargement d'un modèle préentraîné si nécessaire
        if use_pretrained and pretrained_model_name:
            self.load_pretrained(pretrained_model_name)

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

    def load_pretrained(self, model_name: str):
        """
        Charge un modèle préentraîné comme base pour le modèle étudiant.
        :param model_name: Nom du modèle préentraîné (ex: depuis Hugging Face).
        """
        from transformers import AutoModelForCausalLM

        try:
            self.logger.info(
                f"Tentative de chargement du modèle préentraîné {model_name}"
            )
            pretrained_model = AutoModelForCausalLM.from_pretrained(model_name)
            self.token_embedding.weight.data = (
                pretrained_model.get_input_embeddings().weight.data
            )
            self.output_head.weight.data = pretrained_model.lm_head.weight.data
            self.logger.info(f"Modèle préentraîné {model_name} chargé avec succès.")
        except Exception as e:
            self.logger.error(
                f"Erreur lors du chargement du modèle préentraîné {model_name}: {e}"
            )
            raise
