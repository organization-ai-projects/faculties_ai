import torch
from torch import nn

# Définir FLASH_AVAILABLE
try:
    from xformers.ops import memory_efficient_attention

    FLASH_AVAILABLE = True
except ImportError:
    FLASH_AVAILABLE = False


################################################################################
#                          FlashMHA (optionnel)                                #
################################################################################
class FlashMHA(nn.Module):
    """
    Exemple d’utilisation de la xFormers attention (flash-attention).
    Si `FLASH_AVAILABLE` est False, on ne fait rien, on lève une erreur.
    """

    def __init__(self, hidden_size, num_heads, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.dropout = dropout

        head_dim = hidden_size // num_heads
        self.W_q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_v = nn.Linear(hidden_size, hidden_size, bias=False)
        self.proj_out = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x, sin=None, cos=None, rotary=None):
        if not FLASH_AVAILABLE:
            raise ImportError(
                "xFormers ou FlashAttention n'est pas installé. "
                "Installe xformers pour utiliser FlashMHA."
            )
        # x shape: [batch, seq_len, hidden_size]
        b, s, d = x.shape
        head_dim = d // self.num_heads

        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        # Reshape en [batch, seq_len, num_heads, head_dim]
        q = q.view(b, s, self.num_heads, head_dim)
        k = k.view(b, s, self.num_heads, head_dim)
        v = v.view(b, s, self.num_heads, head_dim)

        # Vérification si rotary est utilisé mais sin et cos sont absents
        if rotary is not None:
            if sin is None or cos is None:
                raise ValueError("sin and cos must be provided if rotary is used.")

        # Rotary emb
        if rotary is not None and sin is not None and cos is not None:
            q, k = rotary.apply_rotary(q, k, sin, cos)

        # On fait la attention xFormers
        # reorder => [batch, seq_len, num_heads, head_dim] -> [batch * num_heads, seq_len, head_dim]
        q = q.transpose(1, 2).reshape(b * self.num_heads, s, head_dim)
        k = k.transpose(1, 2).reshape(b * self.num_heads, s, head_dim)
        v = v.transpose(1, 2).reshape(b * self.num_heads, s, head_dim)

        out = memory_efficient_attention(
            q,
            k,
            v,
            p=self.dropout if self.training else 0.0,
            scale=None,
            dropout=True if self.training else False,
        )
        # out: [batch*num_heads, seq_len, head_dim]

        # On retranspose
        out = out.view(b, self.num_heads, s, head_dim).transpose(1, 2).contiguous()
        out = out.view(b, s, d)
        return self.proj_out(out)
