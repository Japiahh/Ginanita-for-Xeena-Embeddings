import torch.nn as nn
from ..config import GinanitaConfig

class AdelinaTokenEmbedding(nn.Module):
    """
    Adelina: Token Embedding (Spesialis Kosakata).
    Mengubah indeks token/kata menjadi vektor padat.
    """
    def __init__(self, config: GinanitaConfig):
        super().__init__()
        # Token embedding menggunakan d_model sebagai dimensi
        self.emb = nn.Embedding(config.vocab_size, config.d_model)
        
    def forward(self, x):
        return self.emb(x)
