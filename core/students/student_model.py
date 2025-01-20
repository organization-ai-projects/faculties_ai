# core/students/student_model.py

import torch
import torch.nn as nn
from core.models.transformer_layer import OptimizedTransformerLayer
from core.models.mlp_mixer import MLPMixer
from core.models.memory_module import MemoryModule
from core.models.cnn import CNN1D
from core.models.rmsnorm import RMSNorm
from core.models.rotary_position_embedding import RotaryPositionEmbedding

class StudentModel(nn.Module):
    def __init__(
        self, 
        vocab_size=50257, 
        hidden_size=768, 
        num_layers=6, 
        num_heads=12, 
        max_seq_len=128, 
        dropout=0.1,
        sd_prob=0.1,
        use_flash_attn=False, 
        ff_multiplier=4
    ):
        super().__init__()

        # Embedding tokens
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.max_seq_len = max_seq_len
        self.hidden_size = hidden_size

        # Rotary Embedding
        self.rotary_emb = RotaryPositionEmbedding(hidden_size // num_heads)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Transformer Layers
        self.transformer_layers = nn.ModuleList([
            OptimizedTransformerLayer(
                hidden_size,
                num_heads,
                dropout=dropout,
                sd_prob=sd_prob * float(layer_idx) / (num_layers - 1) if num_layers > 1 else 0.0,
                use_flash_attn=use_flash_attn,
                ff_multiplier=ff_multiplier
            ) for layer_idx in range(num_layers)
        ])

        # MLP-Mixer for global relationships
        self.mlp_mixer = MLPMixer(hidden_size, max_seq_len)

        # CNN 1D for local patterns
        self.cnn = CNN1D(hidden_size)

        # Memory Module for persistent memory
        self.memory = MemoryModule(hidden_size)

        # Final normalization and output head
        self.final_norm = RMSNorm(hidden_size)
        self.output_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids):
        device = input_ids.device
        batch_size, seq_len = input_ids.shape

        # Embedding tokens
        x = self.token_embedding(input_ids)
        x = self.dropout(x)

        # Compute sin/cos for Rotary
        sin, cos = self.rotary_emb._build_sin_cos(seq_len, device=device)

        # Passing through Transformer layers
        for layer in self.transformer_layers:
            x = layer(x, sin=sin, cos=cos, rotary=self.rotary_emb)

        # MLP-Mixer for global relationships
        x = self.mlp_mixer(x)

        # Passing through CNN 1D for local patterns
        x = x.permute(0, 2, 1)  # [batch, hidden_size, seq_len]
        x = self.cnn(x)
        x = x.permute(0, 2, 1)  # [batch, seq_len, hidden_size]

        # Persistent memory
        x = self.memory(x)

        # Final normalization + output projection
        x = self.final_norm(x)
        logits = self.output_head(x)
        return logits
