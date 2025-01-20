import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    # On essaie d'importer flash-attention ou xFormers
    from xformers.ops import memory_efficient_attention
    FLASH_AVAILABLE = True
except ImportError:
    FLASH_AVAILABLE = False


################################################################################
#                                RMSNorm                                       #
################################################################################
class RMSNorm(nn.Module):
    """
    Implémentation de RMSNorm, alternative à LayerNorm.
    Référence: https://arxiv.org/abs/1910.07467
    """
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # norme RMS = sqrt(mean(x^2))
        norm_x = x.norm(2, dim=-1, keepdim=True)
        rms = norm_x * (1.0 / math.sqrt(x.shape[-1]))
        return self.scale * (x / (rms + self.eps))


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


################################################################################
#                               Rotary Embeddings                              #
################################################################################
class RotaryPositionEmbedding:
    """
    Applique des embeddings rotatifs (RoPE) à Q et K.
    Référence: https://arxiv.org/abs/2104.09864
    """
    def __init__(self, dim_head):
        # On stocke la dimension par tête
        self.dim_head = dim_head

    def _build_sin_cos(self, seq_len, device):
        # On crée des fréquences exponentielles
        # (ex: theta = 10000^( -2i/d ) ) => RoPE style
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.dim_head, 2, device=device).float() / self.dim_head))
        t = torch.arange(seq_len, device=device).float()
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        sin = freqs.sin()
        cos = freqs.cos()
        # shape [seq_len, dim_head/2]
        return sin, cos

    def apply_rotary(self, q, k, sin, cos):
        # q, k: [batch, seq_len, num_heads, dim_head]
        # sin, cos: [seq_len, dim_head/2]
        # On prend la première moitié pour sin, et la seconde pour cos
        # => en pratique, on reshape q en [batch, seq_len, num_heads, dim_head/2, 2]
        # et on applique la rotation
        # Pour simplifier, on peut utiliser la technique de “reshape->rotation->reshape”
        q1 = q[..., : self.dim_head // 2]
        q2 = q[..., self.dim_head // 2 :]
        k1 = k[..., : self.dim_head // 2]
        k2 = k[..., self.dim_head // 2 :]

        # Expand sin, cos sur batch, num_heads
        sin = sin.unsqueeze(1).unsqueeze(2)
        cos = cos.unsqueeze(1).unsqueeze(2)
        # shape => [seq_len, 1, 1, dim_head/2]

        # rotation
        q_rotated = torch.cat([q1 * cos - q2 * sin, q2 * cos + q1 * sin], dim=-1)
        k_rotated = torch.cat([k1 * cos - k2 * sin, k2 * cos + k1 * sin], dim=-1)
        return q_rotated, k_rotated


################################################################################
#                           Stochastic Depth                                   #
################################################################################
def stochastic_depth(x, layer_dropout_prob, mode="row", training=True):
    """
    Applique un “drop” de la sortie de la couche avec une certaine probabilité,
    pour régulariser. Plus la couche est profonde, plus la proba peut être grande.
    """
    if not training or layer_dropout_prob == 0.0:
        return x
    # mode="row" => on génère un masque par batch
    keep_prob = 1.0 - layer_dropout_prob
    size = (x.shape[0], 1, 1) if mode == "row" else x.shape
    mask = torch.rand(size, device=x.device) < keep_prob
    x = x / keep_prob * mask
    return x


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

        # Rotary emb
        if rotary is not None and sin is not None and cos is not None:
            q, k = rotary.apply_rotary(q, k, sin, cos)

        # On fait la attention xFormers
        # reorder => [batch, seq_len, num_heads, head_dim] -> [batch * num_heads, seq_len, head_dim]
        q = q.transpose(1, 2).reshape(b * self.num_heads, s, head_dim)
        k = k.transpose(1, 2).reshape(b * self.num_heads, s, head_dim)
        v = v.transpose(1, 2).reshape(b * self.num_heads, s, head_dim)

        out = memory_efficient_attention(
            q, k, v,
            p = self.dropout if self.training else 0.0,
            scale = None,   # xFormers gère l'échelle
            dropout=True if self.training else False
        )
        # out: [batch*num_heads, seq_len, head_dim]

        # On retranspose
        out = out.view(b, self.num_heads, s, head_dim).transpose(1, 2).contiguous()
        out = out.view(b, s, d)
        return self.proj_out(out)


################################################################################
#                           OptimizedTransformerLayer                          #
################################################################################
class OptimizedTransformerLayer(nn.Module):
    """
    Couche Transformer améliorée :
      - Attention Flash (si dispo) ou nn.MultiheadAttention classique
      - RMSNorm au lieu de LayerNorm
      - Feed Forward GEGLU
      - Stochastic Depth
      - Rotary Embeddings
    """
    def __init__(
        self, 
        hidden_size, 
        num_heads, 
        dropout, 
        sd_prob=0.0, 
        use_flash_attn=False,
        ff_multiplier=4
    ):
        super().__init__()
        self.use_flash_attn = use_flash_attn

        if use_flash_attn and FLASH_AVAILABLE:
            self.attention = FlashMHA(hidden_size, num_heads, dropout)
        else:
            # fallback sur MHA standard
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_size, 
                num_heads=num_heads, 
                dropout=dropout,
                batch_first=True
            )
        self.norm1 = RMSNorm(hidden_size)
        self.feedforward = GatedFeedForward(hidden_size, ff_multiplier=ff_multiplier, dropout=dropout)
        self.norm2 = RMSNorm(hidden_size)

        self.sd_prob = sd_prob  # proba de layer drop

    def forward(self, x, sin=None, cos=None, rotary=None):
        # --- MHA + skip
        normed_x = self.norm1(x)
        if isinstance(self.attention, FlashMHA):
            # FlashMHA
            attn_out = self.attention(normed_x, sin=sin, cos=cos, rotary=rotary)
        else:
            # MultiheadAttention classique
            # On applique rotary sur Q, K manuellement ? On doit ruser un peu.
            # Faute de patch direct, on ne le fait pas. 
            # (Ou on hack MHA en construisant QKV nous-même).
            attn_out, _ = self.attention(normed_x, normed_x, normed_x)

        x = x + stochastic_depth(attn_out, self.sd_prob, training=self.training)

        # --- GEGLU + skip
        normed_x2 = self.norm2(x)
        ff_out = self.feedforward(normed_x2)
        x = x + stochastic_depth(ff_out, self.sd_prob, training=self.training)
        return x


################################################################################
#                                MLPMixer                                      #
################################################################################
class MLPMixer(nn.Module):
    """
    MLP-Mixer pour capturer les relations globales efficacement.
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
        """
        x: [batch, seq_len, hidden_size]
        """
        # Mixer sur la dimension feature
        x = x + self.token_mlp(x)
        # Mixer sur la dimension sequence
        x = x.transpose(1, 2)
        x = x + self.channel_mlp(x)
        x = x.transpose(1, 2)
        return x


################################################################################
#                               MemoryModule                                   #
################################################################################
class MemoryModule(nn.Module):
    """
    Mémoire dynamique pour améliorer la gestion du contexte à long terme.
    """
    def __init__(self, hidden_size, memory_size=256):
        super().__init__()
        self.memory = nn.Parameter(torch.zeros(memory_size, hidden_size))  # Paramètres de mémoire

    def forward(self, x):
        # Ajouter la mémoire persistante à l'entrée
        batch_size, seq_len, hidden_size = x.size()
        memory = self.memory.unsqueeze(0).expand(batch_size, -1, -1)
        return x + memory


################################################################################
#                         OptimizedStudentTransformer                          #
################################################################################
class OptimizedStudentTransformer(nn.Module):
    """
    Architecture hybride optimisée:
      - Embeddings (token + Rotary)
      - Couche Transformer "améliorée" (RMSNorm, GEGLU, Stochastic Depth, FlashAttn option)
      - MLP-Mixer pour relations globales
      - CNN 1D pour motifs locaux
      - Mémoire persistante
    """
    def __init__(
        self, 
        vocab_size=50257, 
        hidden_size=768, 
        num_layers=6, 
        num_heads=12, 
        max_seq_len=128, 
        dropout=0.1,
        sd_prob=0.1,           # prob max pour Stochastic Depth
        use_flash_attn=False,  # active Flash/xFormers si dispo
        ff_multiplier=4
    ):
        super().__init__()

        # Embeddings (token)
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        # On n’utilise plus self.position_embedding, on va faire du rotary
        self.max_seq_len = max_seq_len
        self.hidden_size = hidden_size

        # Rotary
        self.rotary_emb = RotaryPositionEmbedding(dim_head=hidden_size // num_heads)

        self.dropout = nn.Dropout(dropout)

        # Couches Transformer
        self.transformer_layers = nn.ModuleList()
        for layer_idx in range(num_layers):
            # On fait un petit trick: la proba de drop peut augmenter
            # au fur et à mesure des couches (souvent on fait ça).
            layer_sd_prob = sd_prob * float(layer_idx) / (num_layers - 1) if num_layers > 1 else 0.0
            self.transformer_layers.append(
                OptimizedTransformerLayer(
                    hidden_size, 
                    num_heads, 
                    dropout=dropout, 
                    sd_prob=layer_sd_prob,
                    use_flash_attn=use_flash_attn,
                    ff_multiplier=ff_multiplier
                )
            )

        # MLP-Mixer
        self.mlp_mixer = MLPMixer(hidden_size, max_seq_len)

        # CNN 1D
        self.cnn = nn.Conv1d(
            in_channels=hidden_size, 
            out_channels=hidden_size, 
            kernel_size=3, 
            padding=1
        )

        # Mémoire persistante
        self.memory = MemoryModule(hidden_size, memory_size=256)

        # Normalisation finale & tête de sortie
        self.final_norm = RMSNorm(hidden_size)
        self.output_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids):
        """
        input_ids: [batch_size, seq_len]
        """
        device = input_ids.device
        batch_size, seq_len = input_ids.shape

        # Embedding tokens
        x = self.token_embedding(input_ids)
        x = self.dropout(x)

        # Pré-calc sin/cos pour Rotary
        sin, cos = self.rotary_emb._build_sin_cos(seq_len, device=device)

        # Passer par les couches Transformer
        for layer in self.transformer_layers:
            x = layer(x, sin=sin, cos=cos, rotary=self.rotary_emb)

        # MLP-Mixer
        x = self.mlp_mixer(x)

        # CNN 1D pour motifs locaux
        x = x.permute(0, 2, 1)  # [batch, hidden_size, seq_len]
        x = self.cnn(x)
        x = x.permute(0, 2, 1)  # [batch, seq_len, hidden_size]

        # Mémoire persistante
        x = self.memory(x)

        # Normalisation + projection
        x = self.final_norm(x)
        logits = self.output_head(x)
        return logits
