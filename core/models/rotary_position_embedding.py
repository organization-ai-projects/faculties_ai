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
        # => en pratique, on peut utiliser la technique de “reshape->rotation->reshape”
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