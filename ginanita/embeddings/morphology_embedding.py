import torch
import torch.nn as nn
from ..config import GinanitaConfig

class ElinaMorphologyEmbedding(nn.Module):
    def __init__(self, config: GinanitaConfig):
        super().__init__()
        self.emb = nn.Embedding(config.pos_size, config.d_pos)
        nn.init.normal_(self.emb.weight, mean=0.0, std=0.02)

    def forward(self, pos_ids):
        return self.emb(pos_ids)
