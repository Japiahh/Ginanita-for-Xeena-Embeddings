import torch
from ginanita import Ginanita, GinanitaConfig

def main():
    print("Memulai Uji Coba Ginanita...")

    # 1. Konfigurasi
    print("\n[1] Membuat Konfigurasi")
    config = GinanitaConfig(
        vocab_size=5000,
        d_model=256,
        block_size=128,
        feature_dropout=0.2  
    )
    print(f"Config: d_model={config.d_model}, feature_dropout={config.feature_dropout}")
    print(f"Total Concatenated Dimension: {config.total_concat_dim}")

    # 2. Inisialisasi Model
    print("\n[2] Inisialisasi Model")
    model = Ginanita(config)
    print(model)
    print(f"\nTotal Parameter Ginanita: {model.get_num_params():,}")

    # 3. Membuat Dummy Input
    print("\n[3] Membuat Dummy Input (Batch=4, SeqLen=15)")
    B, T = 4, 15
    
    # Simulasikan data token dan tag ID acak
    inputs = {
        'idx': torch.randint(0, config.vocab_size, (B, T)),
        'pos_ids': torch.randint(0, config.pos_size, (B, T)),
        'syn_ids': torch.randint(0, config.syn_size, (B, T)),
        'depth_ids': torch.randint(0, config.max_depth, (B, T)),
        'dep_ids': torch.randint(0, config.dep_size, (B, T)),
        'sent_ids': torch.zeros(B, T, dtype=torch.long)
    }

    # 4. Forward Pass (Training Mode)
    print("\n[4] Forward Pass (Training Mode)")
    model.train()
    output_train = model(**inputs)
    print(f"Output Shape (Train): {output_train.shape}")
    
    # 5. Forward Pass (Eval Mode - No Dropout)
    print("\n[5] Forward Pass (Eval Mode)")
    model.eval()
    with torch.no_grad():
        output_eval = model(**inputs)
    print(f"Output Shape (Eval): {output_eval.shape}")
    
    print("\n[OK] Uji coba berhasil!")

if __name__ == "__main__":
    main()
