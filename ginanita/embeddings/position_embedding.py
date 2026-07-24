import torch
import torch.nn as nn
import math
from ..config import GinanitaConfig

class RayaPositionEmbedding(nn.Module):
    """
    Raya: Menangani Posisi/Urutan.
    Fitur: Sinusoidal Encoding (Fixed) agar bisa menangani kalimat panjang.
    """
    def __init__(self, config: GinanitaConfig):
        super().__init__()
        
        # Matrix Sinusoidal (Fixed), menggunakan config.d_model
        pe = torch.zeros(config.block_size, config.d_model)
        position = torch.arange(0, config.block_size, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, config.d_model, 2).float() * (-math.log(10000.0) / config.d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register sebagai buffer (tidak di-update oleh optimizer)
        self.register_buffer('pe', pe)

    def forward(self, batch_size: int, seq_len: int):
        """
        Args:
            batch_size (int): Ukuran batch dari input saat ini
            seq_len (int): Panjang sekuens input (T)
            
        Returns:
            torch.Tensor: Tensor dengan shape [B, T, d_model]
        """
        # Ambil potongan Positional Encoding sesuai panjang kalimat
        # Shape: [T, d_model]
        pe_slice = self.pe[:seq_len, :]
        
        # Tambah Dimensi Batch: [1, T, d_model]
        pe_slice = pe_slice.unsqueeze(0)
        
        # Expand ke seluruh batch: [B, T, d_model]
        return pe_slice.expand(batch_size, -1, -1)
