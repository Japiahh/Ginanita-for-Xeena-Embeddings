import torch
import torch.nn as nn
import math
from ..config import GinanitaConfig

class RayaPositionEmbedding(nn.Module):
    def __init__(self, config: GinanitaConfig):
        super().__init__()
        pe = torch.zeros(config.block_size, config.d_model)
        position = torch.arange(0, config.block_size, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, config.d_model, 2).float() * (-math.log(10000.0) / config.d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)

    def forward(self, batch_size: int, seq_len: int):
        pe_slice = self.pe[:seq_len, :]
        pe_slice = pe_slice.unsqueeze(0)
        return pe_slice.expand(batch_size, -1, -1)
