import torch
import numpy as np
from model.fastspeech2 import FastSpeech2
from data.dataset import TurkishTTSDataset
from config.model_config import ModelConfig
import argparse
from text import text_to_phonemes  # Türkçe metin-fonem dönüşümü için

def load_checkpoint(checkpoint_path, model, device):
    """
    Checkpoint'i yükler. Farklı checkpoint formatlarını destekler.
    """
    print(f"Checkpoint yükleniyor: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Checkpoint formatını kontrol et
    if isinstance(checkpoint, dict):
        if 'model' in checkpoint:
            # Model durumu 'model' anahtarı altında
            print("'model' anahtarı altında model durumu bulundu")
            model.load_state_dict(checkpoint['model'])
        elif 'model_state_dict' in checkpoint:
            # Model durumu 'model_state_dict' anahtarı altında
            print("'model_state_dict' anahtarı altında model durumu bulundu")
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Doğrudan state_dict
            print("Doğrudan state_dict formatında model durumu bulundu")
            model.load_state_dict(checkpoint)
    else:
        # Doğrudan model durumu
        print("Doğrudan model durumu formatında bulundu")
        model.load_state_dict(checkpoint)
    
    print(f"Model başarıyla yüklendi: {checkpoint_path}")
    return model

def text_to_sequence(text, dataset):
    """Metni fonem ID dizisine çevirir"""
    # Metni fonemlere çevir
    phonemes = text_to_phonemes(text)
    # Fonemleri ID'lere çevir
    phoneme_ids = [dataset.phoneme_to_id[p] for p in phonemes]
    return torch.LongTensor(phoneme_ids)

def generate_default_durations(phoneme_length, mean_duration=5):
    """Basit bir duration tahmini yapar"""
    return torch.ones(phoneme_length) * mean_duration
def print_available_phonemes(dataset):
    """Dataset'teki mevcut fonemleri gösterir"""
    print("\nMevcut fonemler:")
    print(sorted(dataset.phoneme_to_id.keys()))
    print("\nToplam fonem sayısı:", len(dataset.phoneme_to_id))

def text_to_sequence(text, dataset):
    """Metni fonem ID dizisine çevirir"""
    # Önce mevcut fonemleri göster
    print_available_phonemes(dataset)
    
    # Metni fonemlere çevir
    phonemes = text_to_phonemes(text)
    print("\nÇevrilen fonemler:", phonemes)
    
    # Fonemleri ID'lere çevir
    try:
        phoneme_ids = []
        for p in phonemes:
            if p in dataset.phoneme_to_id:
                phoneme_ids.append(dataset.phoneme_to_id[p])
            else:
                print(f"Uyarı: '{p}' fonemi dataset'te bulunamadı, atlanıyor")
                
        if not phoneme_ids:
            raise ValueError("Hiçbir geçerli fonem bulunamadı!")
            
        return torch.LongTensor(phoneme_ids)
    except Exception as e:
        print(f"\nHata: Fonem-ID dönüşümünde sorun: {str(e)}")
        raise

def inference_from_text(args):
    # Cihazı belirle
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Kullanılan cihaz: {device}")

    try:
        # Dataset'i yükle
        print("Dataset yükleniyor...")
        dataset = TurkishTTSDataset(args.csv_path, args.json_path)
        print("Dataset yüklendi")

        # Model oluştur
        print("Model oluşturuluyor...")
        model = FastSpeech2(
            max_seq_len=ModelConfig.max_seq_len,
            phoneme_vocab_size=dataset.get_phoneme_vocab_size(),
            encoder_dim=ModelConfig.encoder_dim,
            encoder_n_layer=ModelConfig.encoder_n_layer,
            encoder_head=ModelConfig.encoder_head,
            encoder_conv1d_filter_size=ModelConfig.encoder_conv1d_filter_size,
            encoder_conv1d_kernel_size=ModelConfig.encoder_conv1d_kernel_size,
            decoder_dim=ModelConfig.decoder_dim,
            decoder_n_layer=ModelConfig.decoder_n_layer,
            decoder_head=ModelConfig.decoder_head,
            decoder_conv1d_filter_size=ModelConfig.decoder_conv1d_filter_size,
            decoder_conv1d_kernel_size=ModelConfig.decoder_conv1d_kernel_size,
            n_mel_channels=ModelConfig.n_mel_channels
        ).to(device)
        print("Model oluşturuldu")

        # Checkpoint'i yükle
        print("Checkpoint yükleniyor...")
        model = load_checkpoint(args.checkpoint_path, model, device)
        model.eval()
        print("Checkpoint yüklendi")

        # Metni fonem dizisine çevir
        print(f"Metin işleniyor: {args.text}")
        phoneme_ids = text_to_sequence(args.text, dataset)
        phoneme_length = torch.tensor([len(phoneme_ids)])
        
        # Basit duration tahmini
        duration = generate_default_durations(len(phoneme_ids))
        mel_length = torch.tensor([duration.sum().item()])

        # Tensörleri GPU'ya taşı ve batch boyutu ekle
        phonemes = phoneme_ids.unsqueeze(0).to(device)
        phoneme_length = phoneme_length.to(device)
        duration = duration.unsqueeze(0).to(device)
        mel_length = mel_length.to(device)

        print(f"Girdi metni: {args.text}")
        print(f"Fonem dizisi uzunluğu: {phoneme_length.item()}")
        print(f"Tahmini mel uzunluğu: {mel_length.item()}")

        # Inference
        print("Mel spektrogram üretiliyor...")
        with torch.no_grad():
            mel_output, _, _, _ = model(
                phonemes,
                phoneme_length,
                duration,
                mel_length
            )

        # Sonuçları kaydet
        mel_output = mel_output.squeeze(0).cpu().numpy()
        np.save(args.output_path, mel_output)
        print(f"Mel spektrogram kaydedildi: {args.output_path}")

        # Görselleştirme
        print("Görsel oluşturuluyor...")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 4))
        plt.imshow(mel_output.T, aspect='auto', origin='lower')
        plt.colorbar()
        plt.title(f'Generated Mel-spectrogram for: {args.text}')
        plt.tight_layout()
        plt.savefig(args.output_path.replace('.npy', '.png'))
        print(f"Görsel kaydedildi: {args.output_path.replace('.npy', '.png')}")

    except Exception as e:
        print(f"Hata oluştu: {str(e)}")
        raise e

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', required=True, help='Model checkpoint dosya yolu')
    parser.add_argument('--csv_path', required=True, help='Dataset CSV dosya yolu')
    parser.add_argument('--json_path', required=True, help='Dataset JSON dosya yolu')
    parser.add_argument('--text', required=True, help='Sentezlenecek metin')
    parser.add_argument('--output_path', default='output_mel.npy', help='Çıktı mel spektrogram dosya yolu')
    
    args = parser.parse_args()
    inference_from_text(args) 