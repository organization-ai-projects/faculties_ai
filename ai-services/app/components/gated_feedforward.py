import torch.nn as nn
################################################################################
#                         Gated Feed Forward (GEGLU)                           #
################################################################################
class GatedFeedForward(nn.Module):
    """
    Remplace la FFN classique par une version GEGLU (Gated GELU).
    Source: https://arxiv.org/abs/2002.05202 (GPT-3 mentionne le GEGLU)
    """
    def __init__(self, hidden_size, ff_multiplier=4, dropout=0.1):
        super().__init__()
        inner_dim = hidden_size * ff_multiplier
        self.w1 = nn.Linear(hidden_size, inner_dim)
        self.w2 = nn.Linear(hidden_size, inner_dim)
        self.act = nn.GELU()
        self.proj = nn.Linear(inner_dim, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Sépare en deux moitiés: la première passera par GELU, l'autre sert de “gate”
        x1 = self.w1(x)
        x2 = self.w2(x)
        # x = GEGLU(x) = (W1*x) ⊗ GELU(W2*x)
        geglu_out = self.act(x2) * x1
        out = self.proj(geglu_out)
        return self.dropout(out)