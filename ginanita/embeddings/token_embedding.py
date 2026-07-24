import torch.nn as nn
from ..config import GinanitaConfig

class AdelinaTokenEmbedding(nn.Module):
    def __init__(self, config: GinanitaConfig):
        super().__init__()
        self.emb = nn.Embedding(config.vocab_size, config.d_model)
        
    def forward(self, x):
        return self.emb(x)
