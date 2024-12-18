class ModelConfig:
    # Model parametreleri
    max_seq_len = 2000
    phoneme_vocab_size = 100  # Türkçe fonem sayınıza göre güncelleyin
    encoder_dim = 256
    encoder_n_layer = 4
    encoder_head = 2
    encoder_conv1d_filter_size = 1024
    encoder_conv1d_kernel_size = 9
    decoder_dim = 256
    decoder_n_layer = 4
    decoder_head = 2
    decoder_conv1d_filter_size = 1024
    decoder_conv1d_kernel_size = 9
    n_mel_channels = 80

    # Eğitim parametreleri
    batch_size = 32
    epochs = 1000
    learning_rate = 0.0005  # Daha düşük learning rate
    min_learning_rate = 1e-5  # Minimum learning rate'i düşür
    warmup_epochs = 5
    save_step = 5
    log_step = 100
    
    # Optimizer parametreleri
    weight_decay = 0.01
    betas = (0.9, 0.98)
    eps = 1e-9
    
    # Scheduler parametreleri
    scheduler_pct_start = 0.1
    
    # Early stopping parametreleri
    patience = 10
    min_improvement = 0.001
    
    # Dropout ve regularizasyon
    dropout = 0.2
    grad_clip_thresh = 0.5  # 1.0'dan düşür

    # Loss ağırlıkları
    mel_weight = 1.0
    duration_weight = 0.05  # 0.1'den düşür
    pitch_weight = 0.05    # 0.1'den düşür
    energy_weight = 0.05   # 0.1'den düşür

    hifigan_config = {
        "resblock": "1",
        "upsample_rates": [8, 8, 2, 2],
        "upsample_kernel_sizes": [16, 16, 4, 4],
        "upsample_initial_channel": 512,
        "resblock_kernel_sizes": [3, 7, 11],
        "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        "n_mel_channels": n_mel_channels,  # FastSpeech2 ile aynı değeri kullan
        "sampling_rate": 24000
    }