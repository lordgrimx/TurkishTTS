import matplotlib
import torch
import numpy as np
import os
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
from pathlib import Path

from model.fastspeech2 import FastSpeech2
from data.dataset import TurkishTTSDataset
from config.model_config import ModelConfig
matplotlib.use('Agg')  # GUI olmadan çalışması için backend'i ayarla
import matplotlib.pyplot as plt
def load_model(checkpoint_path, config, device):
    """Eğitilmiş modeli yükle"""
    # Önce checkpoint'i yükle
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Checkpoint'teki embedding boyutunu al
    embedding_size = checkpoint['model']['embedding.weight'].shape[0]
    print(f"Checkpoint'teki fonem sayısı: {embedding_size}")
    
    # Model konfigürasyonunu güncelle
    config.phoneme_vocab_size = embedding_size
    
    model = FastSpeech2(
        max_seq_len=config.max_seq_len,
        phoneme_vocab_size=config.phoneme_vocab_size,  # Güncellenmiş değer
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
    ).to(device)
    
    # Model ağırlıklarını yükle
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model

def generate_mel_spectrograms(model, dataset, output_dir, device, visualize=True):
    """Veri setindeki her örnek için mel spektrogram üret"""
    mel_dir = os.path.join(output_dir, "mels")
    plot_dir = os.path.join(output_dir, "plots")
    os.makedirs(mel_dir, exist_ok=True)
    
    if visualize:
        os.makedirs(plot_dir, exist_ok=True)
    
    metadata = []
    
    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc="Mel spektrogramlar üretiliyor"):
            # Veri örneğini al
            data = dataset[idx]
            phonemes = data['phonemes'].unsqueeze(0).to(device)
            phoneme_lengths = data['phoneme_length'].unsqueeze(0).to(device)
            
            # Mel spektrogramı üret
            mel_output, *_ = model(
                phonemes,
                phoneme_lengths
            )
            
            # Numpy dizisine çevir
            mel_output = mel_output.squeeze(0).cpu().numpy()
            
            # Mel spektrogramı kaydet
            mel_filename = f"mel_{idx:04d}.npy"
            mel_path = os.path.join(mel_dir, mel_filename)
            np.save(mel_path, mel_output)
            
            # Görselleştirme
            if visualize:
                plot_path = os.path.join(plot_dir, f"mel_{idx:04d}.png")
                plot_mel(
                    mel_output,
                    title=f"Mel Spectrogram {idx}",
                    save_path=plot_path
                )
            
            # Metadata'ya ekle (sadece dosya adını kaydet)
            metadata.append(f"{mel_filename}")
    
    # Metadata'yı kaydet
    metadata_path = os.path.join(output_dir, "metadata.txt")
    with open(metadata_path, "w", encoding="utf-8") as f:
        f.write("\n".join(metadata))
    
    print(f"\nToplam {len(metadata)} mel spektrogram üretildi.")
    print(f"Mel spektrogramlar: {mel_dir}")
    if visualize:
        print(f"Görselleştirmeler: {plot_dir}")
    print(f"Metadata: {metadata_path}")

def plot_mel(mel, title="Mel Spectrogram", save_path=None):
    """Mel spektrogramı görselleştir"""
    plt.figure(figsize=(12, 6))
    plt.imshow(mel, aspect='auto', origin='lower')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.xlabel('Frames')
    plt.ylabel('Mel Channels')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def generate_single_mel(model, text, device, output_dir="output", visualize=True):
    """Tek bir metin için mel spektrogram üret"""
    from text import process_text  # Güncellenmiş text modülünden process_text'i import et
    
    # Çıktı dizinlerini oluştur
    os.makedirs(output_dir, exist_ok=True)
    if visualize:
        plot_dir = os.path.join(output_dir, "plot")
        os.makedirs(plot_dir, exist_ok=True)
    
    # Metni fonem ID'lerine dönüştür
    phoneme_ids = process_text(text, return_ids=True)
    
    # Tensor'lara çevir
    phoneme_tensor = torch.LongTensor(phoneme_ids).unsqueeze(0).to(device)
    phoneme_lengths = torch.LongTensor([len(phoneme_ids)]).to(device)
    
    # Mel spektrogramı üret
    with torch.no_grad():
        mel_output, *_ = model(
            phoneme_tensor,
            phoneme_lengths
        )
    
    # Numpy dizisine çevir
    mel_output = mel_output.squeeze(0).cpu().numpy()
    
    # Mel spektrogramı kaydet
    mel_path = os.path.join(output_dir, "mel_output.npy")
    np.save(mel_path, mel_output)
    
    # Görselleştirme
    if visualize:
        plot_path = os.path.join(plot_dir, "mel_spectrogram.png")
        plot_mel(
            mel_output,
            title=f"Generated Mel Spectrogram for: {text}",
            save_path=plot_path
        )
    
    print(f"\nMel spektrogram üretildi.")
    print(f"Mel spektrogram: {mel_path}")
    if visualize:
        print(f"Görselleştirme: {plot_dir}")
    
    return mel_output

def main(args):
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Kullanılan cihaz: {device}")
    
    # Model yükle
    model = load_model(args.checkpoint_path, ModelConfig, device)
    
    if args.text:
        # Tek bir metin için mel spektrogram üret
        generate_single_mel(model, args.text, device, args.output_dir, args.visualize)
    else:
        # Dataset oluştur ve tüm örnekler için mel spektrogram üret
        dataset = TurkishTTSDataset(
            csv_path=args.csv_path,
            json_path=args.json_path
        )
        generate_mel_spectrograms(model, dataset, args.output_dir, device, args.visualize)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='FastSpeech2 checkpoint dosyasının yolu')
    parser.add_argument('--text', type=str,
                        help='Dönüştürülecek metin')
    parser.add_argument('--csv_path', type=str,
                        help='Veri seti CSV dosyasının yolu')
    parser.add_argument('--json_path', type=str,
                        help='Metadata JSON dosyasının yolu')
    parser.add_argument('--output_dir', type=str, default='generated_mels',
                        help='Üretilen mel spektrogramların kaydedileceği dizin')
    parser.add_argument('--visualize', action='store_true',
                        help='Mel spektrogramları görselleştir')
    
    args = parser.parse_args()
    
    # Argüman kontrolü
    if args.text and (args.csv_path or args.json_path):
        raise ValueError("Ya text parametresi ya da csv/json parametreleri kullanılmalıdır, ikisi birden değil.")
    if not args.text and (not args.csv_path or not args.json_path):
        raise ValueError("Text parametresi verilmediyse csv_path ve json_path zorunludur.")
    
    main(args) 