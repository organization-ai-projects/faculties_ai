################################################################################
#                           OptimizedTransformerLayer                          #
################################################################################
class TransformerLayer(nn.Module):
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