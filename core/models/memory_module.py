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