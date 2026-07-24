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
    def __init__(self, config: GinanitaConfig):
        super().__init__()
        self.config = config

        self.adelina = AdelinaTokenEmbedding(config)
        self.raya = RayaPositionEmbedding(config)
        self.elina = ElinaMorphologyEmbedding(config)
        self.lekha_syn = LekhaSyntacticEmbedding(config)
        self.lekha_dep = LekhaDependencyEmbedding(config)
        self.sent_emb = nn.Embedding(config.max_sentences, config.d_sent)

        self.total_dim = config.total_concat_dim
        self.injector_proj = nn.Linear(self.total_dim, config.d_model)

        self.ln = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        
    @classmethod
    def from_config(cls, config_dict: dict):
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
        B, T = idx.shape

        v_token = self.adelina(idx)
        v_pos   = self.raya(B, T) 
        v_morph = self.elina(pos_ids)
        v_syn   = self.lekha_syn(syn_ids, depth_ids)
        v_dep   = self.lekha_dep(dep_ids)
        v_sent  = self.sent_emb(sent_ids)

        if self.training and self.config.feature_dropout > 0:
            p_keep = 1.0 - self.config.feature_dropout

            mask_morph = torch.bernoulli(torch.full((B, 1, 1), p_keep, device=idx.device))
            mask_syn   = torch.bernoulli(torch.full((B, 1, 1), p_keep, device=idx.device))
            mask_dep   = torch.bernoulli(torch.full((B, 1, 1), p_keep, device=idx.device))

            v_morph = (v_morph * mask_morph) / p_keep
            v_syn   = (v_syn * mask_syn) / p_keep
            v_dep   = (v_dep * mask_dep) / p_keep

        combined = torch.cat([v_token, v_pos, v_morph, v_syn, v_dep, v_sent], dim=-1)

        injected = self.injector_proj(combined)
        return self.dropout(self.ln(injected))
        
    def get_num_params(self) -> int:
        """Hitung jumlah parameter yang membutuhkan gradien."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
