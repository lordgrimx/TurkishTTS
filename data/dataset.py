import torch
import numpy as np
import pandas as pd
import json
from torch.utils.data import Dataset

class TurkishTTSDataset(Dataset):
    def __init__(self, csv_path, json_path):
        # CSV dosyasını yükle
        self.data = pd.read_csv(csv_path)
        
        # JSON metadata'yı yükle
        with open(json_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        
        # Benzersiz fonemleri topla ve fonem-id eşleştirmesi oluştur
        unique_phonemes = set()
        for item in self.metadata:
            unique_phonemes.update(item['phonemes'])
        
        self.phoneme_to_id = {p: i for i, p in enumerate(sorted(unique_phonemes))}
        self.id_to_phoneme = {i: p for p, i in self.phoneme_to_id.items()}

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        item = self.metadata[idx]
        
        # Mel spektrogramı yükle
        mel_path = item['mel_path']
        mel_spectrogram = np.load(mel_path)  # .npy dosyası olduğu için np.load kullanıyoruz
        mel_spectrogram = torch.FloatTensor(mel_spectrogram)
        
        # Fonemleri ID'lere dönüştür
        phonemes = item['phonemes']
        phoneme_ids = [self.phoneme_to_id[p] for p in phonemes]
        phoneme_ids = torch.LongTensor(phoneme_ids)
        
        # Duration'ları tensor'a çevir
        duration = torch.tensor(item['durations'], dtype=torch.long)
        
        return {
            'phonemes': phoneme_ids,
            'mel_spectrogram': mel_spectrogram,
            'duration': duration,
            'phoneme_length': torch.tensor(len(phoneme_ids)),
            'mel_length': torch.tensor(mel_spectrogram.shape[0])
        }

    def get_phoneme_vocab_size(self):
        return len(self.phoneme_to_id)

    def pad_sequence(self, batch):
        # Batch içindeki en uzun sekansın uzunluğunu bul
        max_phoneme_len = max(x['phonemes'].size(0) for x in batch)
        max_mel_len = max(x['mel_spectrogram'].size(0) for x in batch)
        
        # Padding için fonksiyon
        def pad_tensor(tensor, target_len, dim=0):
            pad_size = target_len - tensor.size(dim)
            if pad_size <= 0:
                return tensor
            pad_shape = list(tensor.shape)
            pad_shape[dim] = pad_size
            return torch.cat([tensor, torch.zeros(*pad_shape, dtype=tensor.dtype)], dim=dim)
        
        # Batch'i padding'le
        padded_batch = {
            'phonemes': torch.stack([pad_tensor(x['phonemes'], max_phoneme_len) for x in batch]),
            'mel_spectrogram': torch.stack([pad_tensor(x['mel_spectrogram'], max_mel_len) for x in batch]),
            'duration': torch.stack([pad_tensor(x['duration'], max_phoneme_len) for x in batch]),
            'phoneme_length': torch.stack([x['phoneme_length'] for x in batch]),
            'mel_length': torch.stack([x['mel_length'] for x in batch])
        }
        
        return padded_batch
