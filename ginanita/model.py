import torch
import torch.nn as nn
from typing import Optional

from .config import GinanitaConfig
from .embeddings import (
    AdelinaTokenEmbedding,
    RayaPositionEmbedding,
    ElinaMorphologyEmbedding,
    LekhaSyntacticEmbedding,
    LekhaDependencyEmbedding
)

class Ginanita(nn.Module):
    """
    Ginanita: Multi-Feature Linguistic Injector
    
    Tugas:
    1. Mengumpulkan 6 jenis fitur bahasa (Token, Posisi, POS, Syntax, Depth, Dependency, SentenceID).
    2. Menggabungkan (Concatenate) vektor-vektor tersebut.
    3. Mencampur (Mix/Project) menjadi satu vektor representasi yang padat.
    """
    def __init__(self, config: GinanitaConfig):
        super().__init__()
        self.config = config
        
        # --- 1. TIM SPESIALIS ---
        self.adelina   = AdelinaTokenEmbedding(config)       # Token
        self.raya      = RayaPositionEmbedding(config)       # Posisi
        self.elina     = ElinaMorphologyEmbedding(config)    # POS Tag
        self.lekha_syn = LekhaSyntacticEmbedding(config)     # Syntax + Depth
        self.lekha_dep = LekhaDependencyEmbedding(config)    # Dependency
        self.sent_emb  = nn.Embedding(config.max_sentences, config.d_sent) # Sentence ID
        
        # --- 2. THE MIXER ---
        self.total_dim = config.total_concat_dim
        self.injector_proj = nn.Linear(self.total_dim, config.d_model)
        
        # --- 3. STABILIZER ---
        self.ln = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        
    @classmethod
    def from_config(cls, config_dict: dict):
        """Factory method untuk inisialisasi dari dictionary config."""
        config = GinanitaConfig.from_dict(config_dict)
        return cls(config)

    def forward(
        self, 
        idx: torch.Tensor, 
        pos_ids: torch.Tensor, 
        syn_ids: torch.Tensor, 
        depth_ids: torch.Tensor, 
        dep_ids: torch.Tensor, 
        sent_ids: torch.Tensor
    ):
        """
        Args:
            idx: Token IDs shape [B, T]
            pos_ids: POS tag IDs shape [B, T]
            syn_ids: Syntax label IDs shape [B, T]
            depth_ids: Syntax depth IDs shape [B, T]
            dep_ids: Dependency relation IDs shape [B, T]
            sent_ids: Sentence IDs shape [B, T]
            
        Returns:
            torch.Tensor: Injected features shape [B, T, d_model]
        """
        B, T = idx.shape
        
        # A. AMBIL VEKTOR MENTAH
        v_token = self.adelina(idx)
        v_pos   = self.raya(B, T) # Raya hanya butuh shape
        v_morph = self.elina(pos_ids)
        v_syn   = self.lekha_syn(syn_ids, depth_ids)
        v_dep   = self.lekha_dep(dep_ids)
        v_sent  = self.sent_emb(sent_ids)

        # --- B. FEATURE DROPOUT (Perbaikan Logika) ---
        if self.training and self.config.feature_dropout > 0:
            p_keep = 1.0 - self.config.feature_dropout
            
            # Mask independen untuk masing-masing fitur
            mask_morph = torch.bernoulli(torch.full((B, 1, 1), p_keep, device=idx.device))
            mask_syn   = torch.bernoulli(torch.full((B, 1, 1), p_keep, device=idx.device))
            mask_dep   = torch.bernoulli(torch.full((B, 1, 1), p_keep, device=idx.device))
            
            # Terapkan masker dengan rescaling (/ p_keep)
            v_morph = (v_morph * mask_morph) / p_keep
            v_syn   = (v_syn * mask_syn) / p_keep
            v_dep   = (v_dep * mask_dep) / p_keep
            # v_sent dan v_token tidak di-dropout (fitur utama)

        # C. FUSI (Concatenation)
        combined = torch.cat([v_token, v_pos, v_morph, v_syn, v_dep, v_sent], dim=-1)
        
        # D. INJECTION & FINISH
        injected = self.injector_proj(combined)
        return self.dropout(self.ln(injected))
        
    def get_num_params(self) -> int:
        """Hitung jumlah parameter yang membutuhkan gradien."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
