import torch
from typing import Dict, Any

def preprocess_sentence(sentence_data: Dict[str, Any], mappings: Dict[str, Dict[str, int]], device: str = 'cpu') -> Dict[str, torch.Tensor]:
    tokens = sentence_data.get('token', [])
    T = len(tokens)
    
    inputs = {
        'idx': torch.zeros(1, T, dtype=torch.long, device=device),
        'pos_ids': torch.zeros(1, T, dtype=torch.long, device=device),
        'syn_ids': torch.zeros(1, T, dtype=torch.long, device=device),
        'depth_ids': torch.zeros(1, T, dtype=torch.long, device=device),
        'dep_ids': torch.zeros(1, T, dtype=torch.long, device=device),
        'sent_ids': torch.zeros(1, T, dtype=torch.long, device=device),
    }
    
    return inputs
