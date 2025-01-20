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