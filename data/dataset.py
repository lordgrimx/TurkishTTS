import torch
import numpy as np
import pandas as pd
import json
import torchaudio
from torch.utils.data import Dataset
import os
import random
import torch.nn.functional as F

from config.model_config import ModelConfig

class TurkishTTSDataset(Dataset):
    def __init__(self, csv_path, json_path, max_wav_length=160000, training=True):  # ~10 saniye
        self.training = training  # training modunu sakla
        self.max_wav_length = max_wav_length
        self.base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # TTSModel dizini
        
        # Dosyaların varlığını kontrol et
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV dosyası bulunamadı: {csv_path}")
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"JSON dosyası bulunamadı: {json_path}")
        
        # Metadata'yı yükle
        self.data = pd.read_csv(csv_path)
        
        with open(json_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        
        # Ses dosyası yollarını güncelle
        self.data['path'] = self.data['path'].apply(
            lambda x: os.path.normpath(os.path.join(self.base_path, 'data', x.replace('/', os.sep)))
        )
        
        # Mel dosyalarının yolunu güncelle
        for item in self.metadata:
            # Eğer mel_path zaten "data/mels" ile başlamıyorsa, güncelle
            if not item['mel_path'].startswith('data/mels'):
                item['mel_path'] = os.path.join('data', item['mel_path'])
        
        # Mel dosyalarının varlığını kontrol et
        missing_mels = []
        for item in self.metadata:
            if not os.path.exists(item['mel_path']):
                missing_mels.append(item['mel_path'])
        
        if missing_mels:
            print(f"Uyarı: {len(missing_mels)} mel dosyası bulunamadı.")
            print("İlk 5 eksik dosya:", missing_mels[:5])
        
        # Benzersiz fonemleri topla ve fonem-id eşleştirmesi oluştur
        unique_phonemes = set()
        for item in self.metadata:
            unique_phonemes.update(item['phonemes'])
        
        self.phoneme_to_id = {p: i for i, p in enumerate(sorted(unique_phonemes))}
        self.id_to_phoneme = {i: p for p, i in self.phoneme_to_id.items()}

    def __getitem__(self, idx):
        item = self.metadata[idx]
        
        try:
            # Mel spektrogramı numpy array olarak yükle ve torch tensor'a çevir
            mel_path = os.path.join(self.base_path, self.metadata[idx]['mel_path'])
            try:
                mel_spectrogram = np.load(mel_path)
                mel_spectrogram = torch.FloatTensor(mel_spectrogram)
            except Exception as e:
                print(f"Mel dosyası yüklenirken hata: {mel_path}")
                print(f"Hata: {str(e)}")
                raise
            
            # Mel spektrogramı yükle ve boyutunu kontrol et
            mel_spectrogram = np.load(mel_path)
            mel_spectrogram = torch.FloatTensor(mel_spectrogram)
            
            # Mel boyutunu kontrol et ve düzelt
            if mel_spectrogram.size(1) != ModelConfig.n_mel_channels:
                mel_spectrogram = F.interpolate(
                    mel_spectrogram.unsqueeze(0).unsqueeze(0),
                    size=(mel_spectrogram.size(0), ModelConfig.n_mel_channels),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0).squeeze(0)
            
            # Boyut kontrolü
            assert mel_spectrogram.size(1) == ModelConfig.n_mel_channels, \
                f"Mel spektrogram yanlış boyutta: {mel_spectrogram.size()}"
            
            # Varsayılan değerler
            waveform = torch.zeros((1, self.max_wav_length))
            sr = 16000
            
            # Waveform yükleme işlemi
            try:
                wav_path = self.data.iloc[idx]['path']
                wav_path = os.path.normpath(wav_path)
                if os.path.exists(wav_path):
                    waveform, sr = torchaudio.load(wav_path)
                    
                    if sr != 16000:
                        resampler = torchaudio.transforms.Resample(sr, 16000)
                        waveform = resampler(waveform)
                    
                    if waveform.size(1) > self.max_wav_length:
                        waveform = waveform[:, :self.max_wav_length]
                    elif waveform.size(1) < self.max_wav_length:
                        padding = self.max_wav_length - waveform.size(1)
                        waveform = torch.nn.functional.pad(waveform, (0, padding))
            
            except Exception as e:
                pass  # Ses dosyası yüklenemezse varsayılan waveform kullan
            
            # Fonemleri ID'lere dönüştür
            phonemes = item['phonemes']
            phoneme_ids = [self.phoneme_to_id[p] for p in phonemes]
            phoneme_ids = torch.LongTensor(phoneme_ids)
            
            # Duration'ları tensor'a çevir
            duration = torch.tensor(item['durations'], dtype=torch.long)
            
            # Data augmentation
            if self.training:
                if random.random() < 0.5:
                    pitch_shift = random.uniform(-2, 2)
                    mel_spectrogram = mel_spectrogram + pitch_shift
                
                if random.random() < 0.5:
                    stretch_factor = random.uniform(0.9, 1.1)
                    mel_spectrogram = F.interpolate(
                        mel_spectrogram.unsqueeze(0),
                        scale_factor=stretch_factor,
                        mode='linear',
                        align_corners=False
                    ).squeeze(0)
            
            return {
                'phonemes': torch.LongTensor(phoneme_ids),
                'mel_spectrogram': mel_spectrogram,
                'duration': torch.LongTensor(duration),
                'phoneme_length': torch.tensor(len(phoneme_ids)),
                'mel_length': torch.tensor(mel_spectrogram.size(0)),
                'waveform': waveform
            }
            
        except Exception as e:
            print(f"Hata idx {idx} için: {str(e)}")
            print(f"Mel path: {mel_path}")
            raise

    def collate_fn(self, batch):
        # En uzun dizilerin uzunluklarını bul
        max_phoneme_len = max([len(x['phonemes']) for x in batch])
        max_mel_len = max([x['mel_spectrogram'].size(0) for x in batch])
        max_wave_len = max([x['waveform'].size(1) for x in batch])
        
        def pad_and_resize(mel_spec):
            # Mel spektrogramı doğru boyuta getir
            if mel_spec.size(1) != ModelConfig.n_mel_channels:
                # Doğru boyutlara getir
                mel_spec = mel_spec.unsqueeze(0).unsqueeze(0)  # [1, 1, T, F]
                mel_spec = F.interpolate(
                    mel_spec,
                    size=(mel_spec.size(2), ModelConfig.n_mel_channels),
                    mode='bilinear',
                    align_corners=False
                )  # [1, 1, T, 80]
                mel_spec = mel_spec.squeeze(0).squeeze(0)  # [T, 80]
            
            # Zaman boyutunda padding uygula
            if mel_spec.size(0) < max_mel_len:
                mel_spec = F.pad(mel_spec, (0, 0, 0, max_mel_len - mel_spec.size(0)))
            elif mel_spec.size(0) > max_mel_len:
                mel_spec = mel_spec[:max_mel_len, :]
            
            return mel_spec

        def pad_tensor(vec, pad_len):
            if len(vec.shape) == 1:  # 1D tensor (phonemes, duration)
                return F.pad(vec, (0, pad_len - vec.shape[0]))
            else:  # waveform için
                return F.pad(vec, (0, pad_len - vec.shape[1]))

        try:
            # Her bir mel spektrogramı önce doğru boyuta getir
            resized_mels = []
            for i, x in enumerate(batch):
                mel = x['mel_spectrogram']
                try:
                    resized_mel = pad_and_resize(mel)
                    
                    # Boyut kontrolü
                    if resized_mel.size(1) != ModelConfig.n_mel_channels:
                        print(f"Mel {i} boyutu düzeltiliyor: {mel.size()} -> {resized_mel.size()}")
                        # Son bir deneme daha
                        resized_mel = F.interpolate(
                            resized_mel.unsqueeze(0).unsqueeze(0),
                            size=(resized_mel.size(0), ModelConfig.n_mel_channels),
                            mode='bilinear',
                            align_corners=False
                        ).squeeze(0).squeeze(0)
                    
                    resized_mels.append(resized_mel)
                except Exception as e:
                    print(f"Mel {i} işlenirken hata: {str(e)}")
                    print(f"Orijinal boyut: {mel.size()}")
                    raise
            
            padded_batch = {
                'phonemes': torch.stack([pad_tensor(x['phonemes'], max_phoneme_len) for x in batch]),
                'mel_spectrogram': torch.stack(resized_mels),
                'duration': torch.stack([pad_tensor(x['duration'], max_phoneme_len) for x in batch]),
                'phoneme_length': torch.tensor([len(x['phonemes']) for x in batch]),
                'mel_length': torch.tensor([x['mel_spectrogram'].size(0) for x in batch]),
                'waveform': torch.stack([pad_tensor(x['waveform'], max_wave_len) for x in batch])
            }
            return padded_batch
        
        except Exception as e:
            print("Collate hatası:")
            print(f"Batch boyutları: {[x['mel_spectrogram'].shape for x in batch]}")
            print(f"Hata: {str(e)}")
            raise

    def __len__(self):
        return len(self.metadata)

    def get_phoneme_vocab_size(self):
        return len(self.phoneme_to_id)

    
