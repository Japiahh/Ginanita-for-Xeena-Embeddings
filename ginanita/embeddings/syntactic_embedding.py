import torch
import torch.nn as nn
from ..config import GinanitaConfig

class LekhaSyntacticEmbedding(nn.Module):
    """
    Lekha (Syntactic): Syntax Tree Label & Depth Embedding.
    Menangkap struktur hirarkis kalimat (NP, VP, dll.) beserta kedalamannya.
    """
    def __init__(self, config: GinanitaConfig):
        super().__init__()
        # Syntax Label (NP/VP) -> Dimensi Kecil (d_syn)
        self.syn_emb = nn.Embedding(config.syn_size, config.d_syn)
        
        # Depth Level -> Dimensi Sangat Kecil (d_depth)
        self.depth_emb = nn.Embedding(config.max_depth, config.d_depth) 

    def forward(self, syn_ids, depth_ids):
        # Gabungkan Label dan Depth di fitur terakhir
        v_syn = self.syn_emb(syn_ids)
        v_depth = self.depth_emb(depth_ids)
        
        # Return gabungan: shape [B, T, d_syn + d_depth]
        return torch.cat([v_syn, v_depth], dim=-1)
