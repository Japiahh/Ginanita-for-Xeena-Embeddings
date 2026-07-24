import torch.nn as nn
from ..config import GinanitaConfig

class LekhaDependencyEmbedding(nn.Module):
    """
    Lekha (Dependency): Dependency Relation Embedding.
    Menangkap hubungan dependensi antar kata (root, nsubj, obj).
    """
    def __init__(self, config: GinanitaConfig):
        super().__init__()
        self.emb = nn.Embedding(config.dep_size, config.d_dep)
        
    def forward(self, x): 
        return self.emb(x)
