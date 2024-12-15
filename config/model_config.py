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
    batch_size = 16
    epochs = 20
    learning_rate = 0.0001
    save_step = 5
    log_step = 100