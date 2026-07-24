import dataclasses
from dataclasses import dataclass, field
import json

@dataclass
class GinanitaConfig:
    """
    Konfigurasi untuk modul Ginanita.
    Mengatur semua dimensi embedding, jumlah label, dan parameter regularisasi.
    """
    # --- Core Parameters ---
    vocab_size: int = 10000
    d_model: int = 512          # Dimensi output akhir & token embedding utama
    block_size: int = 256       # Panjang sequence maksimal
    
    # --- Fitur Linguistik (Spesialis) ---
    pos_size: int = 50          # Jumlah POS tags (Morphology)
    d_pos: int = 32             # Dimensi POS/morphology embedding
    
    syn_size: int = 30          # Jumlah syntax labels
    d_syn: int = 24             # Dimensi syntax label embedding
    max_depth: int = 20         # Kedalaman maks syntax tree
    d_depth: int = 16           # Dimensi depth embedding
    
    dep_size: int = 20          # Jumlah dependency relations
    d_dep: int = 24             # Dimensi dependency embedding
    
    max_sentences: int = 16     # Maksimal kalimat dalam satu input sequence
    d_sent: int = 16            # Dimensi sentence ID embedding
    
    # --- Regularisasi ---
    dropout: float = 0.1        # Dropout standar setelah injector projection
    feature_dropout: float = 0.0 # Probabilitas fitur pendukung didropout saat training (0 = disabled)
    
    def __post_init__(self):
        """Validasi parameter setelah inisialisasi."""
        assert self.vocab_size > 0
        assert self.d_model > 0
        assert self.block_size > 0
        assert 0.0 <= self.dropout < 1.0
        assert 0.0 <= self.feature_dropout < 1.0

    @property
    def total_concat_dim(self) -> int:
        """Total dimensi setelah semua embedding di-concatenate."""
        # Token (d_model) + Posisi (d_model) + Morphology (d_pos) + 
        # Syntax (d_syn) + Depth (d_depth) + Dependency (d_dep) + SentenceID (d_sent)
        return (self.d_model * 2) + self.d_pos + self.d_syn + self.d_depth + self.d_dep + self.d_sent

    def to_dict(self):
        """Ubah config menjadi dictionary."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d):
        """Buat config dari dictionary."""
        return cls(**d)
    
    def save_json(self, file_path: str):
        """Simpan config ke file JSON."""
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, file_path: str):
        """Muat config dari file JSON."""
        with open(file_path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))
