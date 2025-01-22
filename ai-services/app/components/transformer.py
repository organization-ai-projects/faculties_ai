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
        self.memory = MemoryModule(hidden_size)

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
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)

        # Mémoire persistante
        x = self.memory(x)

        # Normalisation + projection
        x = self.final_norm(x)
        logits = self.output_head(x)
        return logits
