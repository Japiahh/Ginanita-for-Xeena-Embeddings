import dataclasses
from dataclasses import dataclass, field
import json

@dataclass
class GinanitaConfig:
    vocab_size: int = 10000
    d_model: int = 512
    block_size: int = 256

    pos_size: int = 50
    d_pos: int = 32
    
    syn_size: int = 30
    d_syn: int = 24
    max_depth: int = 20
    d_depth: int = 16
    
    dep_size: int = 20
    d_dep: int = 24
    
    max_sentences: int = 16
    d_sent: int = 16

    dropout: float = 0.1
    feature_dropout: float = 0.0
    
    def __post_init__(self):
        assert self.vocab_size > 0
        assert self.d_model > 0
        assert self.block_size > 0
        assert 0.0 <= self.dropout < 1.0
        assert 0.0 <= self.feature_dropout < 1.0

    @property
    def total_concat_dim(self) -> int:
        return (self.d_model * 2) + self.d_pos + self.d_syn + self.d_depth + self.d_dep + self.d_sent

    def to_dict(self):
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**d)
    
    def save_json(self, file_path: str):
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, file_path: str):
        with open(file_path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))
