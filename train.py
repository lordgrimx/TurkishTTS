import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F 
import numpy as np
import os
import argparse
from tqdm import tqdm
from torch.amp import autocast, GradScaler
import shutil
import time
import tempfile
import matplotlib.pyplot as plt

from model.fastspeech2 import FastSpeech2
from data.dataset import TurkishTTSDataset  # Dataset sınıfımızı import ediyoruz
from utils.loss import FastSpeech2Loss
from config.model_config import ModelConfig

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # TensorFlow log seviyesini azalt
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # oneDNN mesajlarını kapat

def calculate_mel_accuracy(pred_mel, target_mel, threshold=0.1):
    """Mel spektrogramlar arasındaki benzerliği hesaplar"""
    with torch.no_grad():
        # Mutlak fark
        diff = torch.abs(pred_mel - target_mel)
        # Threshold'a göre doğru tahmin edilen değerlerin oranı
        correct = (diff < threshold).float().mean()
        return correct.item() * 100

def save_checkpoint(model, optimizer, scaler, epoch, path):
    """
    Model checkpoint'ini kaydeder
    """
    print(f"Checkpoint kaydediliyor: {path}")
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'scaler': scaler.state_dict() if scaler is not None else None,
    }, path)
    print(f"Checkpoint kaydedildi: {path}")

def main(args):
    # Geçici bir dizin oluştur
    temp_dir = tempfile.mkdtemp()
    log_dir = os.path.join(temp_dir, 'logs')

    print(f"Geçici log dizini: {log_dir}")

    try:
        os.makedirs(log_dir, exist_ok=True)
        time.sleep(1)  # Dizinin oluşturulması için bekle
        
        # Dizin izinlerini kontrol et
        print(f"Dizin yazılabilir mi?: {os.access(log_dir, os.W_OK)}")
        print(f"Dizin okunabilir mi?: {os.access(log_dir, os.R_OK)}")
        
        writer = SummaryWriter(log_dir=log_dir)
    except Exception as e:
        print(f"Writer oluşturulurken hata: {e}")
        print(f"Log dizini mevcut mu?: {os.path.exists(log_dir)}")
        print(f"Log dizini bir dizin mi?: {os.path.isdir(log_dir)}")
        # Dizindeki dosyaları listele
        if os.path.exists(log_dir):
            print(f"Dizin içeriği: {os.listdir(log_dir)}")
        raise
    
    # Dizinlerin var olduğundan emin ol
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Device configuration - GPU kontrolü ekleyelim
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Kullanılan cihaz: {device}")
    if torch.cuda.is_available():
        print(f"GPU adı: {torch.cuda.get_device_name()}")
    
    print(f"CUDA kullanılabilir mi?: {torch.cuda.is_available()}")
    print(f"Mevcut CUDA cihaz sayısı: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"Şu anki CUDA cihazı: {torch.cuda.current_device()}")
        print(f"CUDA cihaz adı: {torch.cuda.get_device_name()}")
    
    # Mixed precision için GradScaler
    scaler = GradScaler()
    
    # Dataset ve DataLoader oluşturma
    dataset = TurkishTTSDataset(
        csv_path=args.csv_path,
        json_path=args.json_path
    )
    
    train_loader = DataLoader(
        dataset, 
        batch_size=ModelConfig.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=dataset.pad_sequence  # Collate fonksiyonunu ekledik
    )
    
    # Model
    model = FastSpeech2(
        max_seq_len=ModelConfig.max_seq_len,
        phoneme_vocab_size=dataset.get_phoneme_vocab_size(),  # Değiştirildi
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
    ).to(device)  # Model'i GPU'ya taşı
    
    # Loss function tanımla
    criterion = FastSpeech2Loss()
    
    # Optimizer ve scheduler tanımlamaları
    optimizer = torch.optim.AdamW(model.parameters(), lr=ModelConfig.learning_rate)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=ModelConfig.learning_rate,
        epochs=ModelConfig.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
        anneal_strategy='cos'
    )

    
    # Training loop
    for epoch in range(1, ModelConfig.epochs + 1):
        model.train()
        total_loss = 0
        total_accuracy = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}')
        for i, batch in enumerate(progress_bar):
            optimizer.zero_grad()
            
            # Batch verilerini al
            phonemes = batch['phonemes'].to(device)
            mel_target = batch['mel_spectrogram'].to(device)
            duration = batch['duration'].to(device)
            phoneme_length = batch['phoneme_length'].to(device)
            mel_length = batch['mel_length'].to(device)
            
            # Forward pass
            with autocast(device_type='cuda'):
                mel_output, _, _, _ = model(
                    phonemes,
                    phoneme_length,
                    duration,
                    mel_length
                )
                
                # Mel spektrogramları aynı boyuta getir
                max_len = mel_target.size(1)
                if mel_output.size(1) > max_len:
                    mel_output = mel_output[:, :max_len, :]
                elif mel_output.size(1) < max_len:
                    mel_output = F.pad(mel_output, (0, 0, 0, max_len - mel_output.size(1)))
                
                loss = criterion(mel_output, mel_target)
                
                # Accuracy hesapla
                accuracy = calculate_mel_accuracy(mel_output, mel_target)
                total_accuracy += accuracy
            
            # Backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            avg_loss = total_loss / (i + 1)
            avg_accuracy = total_accuracy / (i + 1)
            
            # Progress bar güncelle
            progress_bar.set_description(
                f'Epoch {epoch}, Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.2f}%'
            )
            
            # Tensorboard'a logla
            step = (epoch - 1) * len(train_loader) + i
            writer.add_scalar('Loss/train', loss.item(), step)
            writer.add_scalar('Accuracy/train', accuracy, step)
            
           
        
        # Epoch sonunda checkpoint kaydet
        if epoch % ModelConfig.save_step == 0:
            save_checkpoint(
                model, optimizer, scaler, epoch,
                os.path.join(args.checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
            )
            
        # Learning rate'i güncelle
        scheduler.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, required=True,
                        help='path to preprocessed csv file')
    parser.add_argument('--json_path', type=str, required=True,
                        help='path to metadata json file')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='directory to save tensorboard logs')
    
    args = parser.parse_args()
    
    # Create directories if not exist
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    main(args)