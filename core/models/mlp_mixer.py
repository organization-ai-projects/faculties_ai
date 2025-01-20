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