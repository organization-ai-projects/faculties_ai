import torch
import torch.nn as nn


class MLPMixer(nn.Module):
    """
    MLP-Mixer for efficiently capturing global relationships.
    """

    def __init__(self, hidden_size, seq_len):
        super().__init__()
        self.mlp1 = nn.Linear(seq_len, seq_len)
        self.mlp2 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        x: [batch, seq_len, hidden_size]
        """
        # Token mixing
        x_t = x.transpose(1, 2)
        x_t = self.mlp1(x_t)
        x_t = self.relu(x_t)
        x = x_t.transpose(1, 2)

        # Channel mixing
        x = self.mlp2(x)
        x = self.relu(x)
        return x
