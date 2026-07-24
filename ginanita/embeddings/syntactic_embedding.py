import torch
import torch.nn as nn
from ..config import GinanitaConfig

class LekhaSyntacticEmbedding(nn.Module):
    def __init__(self, config: GinanitaConfig):
        super().__init__()
        self.syn_emb = nn.Embedding(config.syn_size, config.d_syn)
        self.depth_emb = nn.Embedding(config.max_depth, config.d_depth) 
        
    def forward(self, syn_ids, depth_ids):
        v_syn = self.syn_emb(syn_ids)
        v_depth = self.depth_emb(depth_ids)
        
        return torch.cat([v_syn, v_depth], dim=-1)
