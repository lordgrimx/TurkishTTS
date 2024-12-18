# model/tts_pipeline.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace

from .fastspeech2 import FastSpeech2
from hifi_gan.models import Generator as HifiGAN

class TTSPipeline(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # FastSpeech2 model
        self.fastspeech2 = FastSpeech2(
            max_seq_len=config.max_seq_len,
            phoneme_vocab_size=config.phoneme_vocab_size,
            encoder_dim=config.encoder_dim,
            encoder_n_layer=config.encoder_n_layer,
            encoder_head=config.encoder_head,
            encoder_conv1d_filter_size=config.encoder_conv1d_filter_size,
            encoder_conv1d_kernel_size=config.encoder_conv1d_kernel_size,
            decoder_dim=config.decoder_dim,
            decoder_n_layer=config.decoder_n_layer,
            decoder_head=config.decoder_head,
            decoder_conv1d_filter_size=config.decoder_conv1d_filter_size,
            decoder_conv1d_kernel_size=config.decoder_conv1d_kernel_size,
            n_mel_channels=config.n_mel_channels
        )
        
        # HiFi-GAN model configuration
        hifigan_config = {
            "resblock": "1",
            "num_gpus": 0,
            "batch_size": 16,
            "learning_rate": 0.0002,
            "adam_b1": 0.8,
            "adam_b2": 0.99,
            "lr_decay": 0.999,
            "seed": 1234,
            "upsample_rates": [8,8,2,2],
            "upsample_kernel_sizes": [16,16,4,4],
            "upsample_initial_channel": 512,
            "resblock_kernel_sizes": [3,7,11],
            "resblock_dilation_sizes": [[1,3,5], [1,3,5], [1,3,5]],
            "segment_size": 8192,
            "num_mels": 80,
            "num_freq": 1025,
            "n_fft": 1024,
            "hop_size": 256,
            "win_size": 1024,
            "sampling_rate": 22050,
            "fmin": 0,
            "fmax": 8000
        }
        
        # Dict'i SimpleNamespace'e çevir
        hifigan_config = SimpleNamespace(**hifigan_config)
        
        # HiFi-GAN model
        self.hifigan = HifiGAN(hifigan_config)
        
    def forward(self, phonemes, phoneme_lengths, durations=None, mel_lengths=None):
        """
        Args:
            phonemes (Tensor): Fonem dizisi [batch_size, max_phoneme_len]
            phoneme_lengths (Tensor): Fonem uzunlukları [batch_size]
            durations (Tensor, optional): Süre tahminleri [batch_size, max_phoneme_len]
            mel_lengths (Tensor, optional): Mel spektrogram uzunlukları [batch_size]
            
        Returns:
            Tuple[Tensor, Tensor]: (ses dalgası, mel spektrogram)
        """
        # FastSpeech2 ile mel spektrogram üret
        mel_output, *_ = self.fastspeech2(
            phonemes,
            phoneme_lengths,
            durations,
            mel_lengths
        )
        
        # HiFi-GAN için mel spektrogramı düzenle
        mel_for_hifigan = mel_output.transpose(1, 2)
        
        # HiFi-GAN ile ses dalgası üret
        audio_output = self.hifigan(mel_for_hifigan)
        
        # Orijinal mel_output formatını koru
        return audio_output, mel_output  # mel_output'u transpose edilmemiş halde döndür
    
    def inference(self, phonemes):
        """
        Tek bir metin için çıkarım yapar
        
        Args:
            phonemes (Tensor): Fonem dizisi [1, phoneme_len]
            
        Returns:
            Tuple[Tensor, Tensor]: (ses dalgası, mel spektrogram)
        """
        # Cihazı kontrol et
        device = next(self.parameters()).device
        phonemes = phonemes.to(device)
        
        # Batch boyutu 1 için uzunluk tensörü oluştur
        phoneme_lengths = torch.LongTensor([phonemes.size(1)]).to(device)
        
        # Değerlendirme moduna geç
        self.eval()
        
        with torch.no_grad():
            # FastSpeech2 ile mel spektrogram üret
            mel_output, *_ = self.fastspeech2(
                phonemes,
                phoneme_lengths
            )
            
            # HiFi-GAN ile ses dalgası üret
            audio_output = self.hifigan(mel_output)
        
        return audio_output, mel_output
    
    def get_mel_spectrogram(self, phonemes, phoneme_lengths, durations=None, mel_lengths=None):
        """
        Sadece mel spektrogram üretir
        
        Args:
            phonemes (Tensor): Fonem dizisi [batch_size, max_phoneme_len]
            phoneme_lengths (Tensor): Fonem uzunlukları [batch_size]
            durations (Tensor, optional): Süre tahminleri [batch_size, max_phoneme_len]
            mel_lengths (Tensor, optional): Mel spektrogram uzunlukları [batch_size]
            
        Returns:
            Tensor: Mel spektrogram [batch_size, n_mel_channels, mel_len]
        """
        mel_output, *_ = self.fastspeech2(
            phonemes,
            phoneme_lengths,
            durations,
            mel_lengths
        )
        return mel_output
    
    def generate_audio(self, mel_spectrogram):
        """
        Mel spektrogramdan ses dalgası üretir
        
        Args:
            mel_spectrogram (Tensor): Mel spektrogram [batch_size, n_mel_channels, mel_len]
            
        Returns:
            Tensor: Ses dalgası [batch_size, 1, wave_len]
        """
        return self.hifigan(mel_spectrogram)