import torch
import torch.nn as nn

class HybridTransformer(nn.Module):
    """
    Architecture hybride : Transformer amélioré avec MLP-Mixer, CNN et mémoire persistante.
    """
    def __init__(self, vocab_size=50257, hidden_size=768, num_layers=6, num_heads=12, max_seq_len=128, dropout=0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(max_seq_len, hidden_size)
        self.dropout = nn.Dropout(dropout)

        # Transformer amélioré
        self.transformer_layers = nn.ModuleList([
            OptimizedTransformerLayer(hidden_size, num_heads, dropout)
            for _ in range(num_layers)
        ])

        # MLP-Mixer
        self.mlp_mixer = MLPMixer(hidden_size, max_seq_len)

        # CNN 1D pour les motifs locaux
        self.cnn = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=3, padding=1)

        # Mémoire persistante
        self.memory = MemoryModule(hidden_size, max_memory_size=64)

        # Normalisation et sortie
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.output_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids):
        seq_len = input_ids.size(1)
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

        # Embeddings
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.dropout(x)

        # Passer par les couches Transformer
        for layer in self.transformer_layers:
            x = layer(x)

        # MLP-Mixer pour la fusion globale
        x = self.mlp_mixer(x)

        # CNN pour capturer les motifs locaux
        x = x.permute(0, 2, 1)  # Passer à [batch, hidden_size, seq_len] pour CNN
        x = self.cnn(x)
        x = x.permute(0, 2, 1)  # Retour à [batch, seq_len, hidden_size]

        # Mémoire persistante
        x = self.memory(x)

        # Normalisation et sortie
        x = self.layer_norm(x)
        logits = self.output_head(x)
        return logits


class OptimizedTransformerLayer(nn.Module):
    """
    Une couche Transformer avec Flash Attention et optimisations.
    """
    def __init__(self, hidden_size, num_heads, dropout):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.feedforward = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = x + self.dropout(self.norm1(attn_output))
        ff_output = self.feedforward(x)
        x = x + self.dropout(self.norm2(ff_output))
        return x


class MLPMixer(nn.Module):
    """
    MLP-Mixer pour capturer les relations globales.
    """
    def __init__(self, hidden_size, seq_len):
        super().__init__()
        self.token_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size)
        )
        self.channel_mlp = nn.Sequential(
            nn.Linear(seq_len, seq_len),
            nn.GELU(),
            nn.Linear(seq_len, seq_len)
        )

    def forward(self, x):
        # Mixer sur les dimensions des tokens
        x = x + self.token_mlp(x)
        # Mixer sur les dimensions des canaux
        x = x.transpose(1, 2)
        x = x + self.channel_mlp(x)
        x = x.transpose(1, 2)
        return x


class MemoryModule(nn.Module):
    """
    Mémoire persistante pour améliorer le contexte.
    """
    def __init__(self, hidden_size, max_memory_size=64):
        super().__init__()
        self.memory = nn.Parameter(torch.zeros(max_memory_size, hidden_size))

    def forward(self, x):
        # Ajouter la mémoire à l'entrée
        batch_size, seq_len, hidden_size = x.size()
        memory = self.memory.unsqueeze(0).expand(batch_size, -1, -1)
        return x + memory
