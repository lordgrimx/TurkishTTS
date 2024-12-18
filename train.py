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
import matplotlib
matplotlib.use('Agg')  # GUI olmayan backend'i kullan
import matplotlib.pyplot as plt
from torch.utils.data import random_split

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

def calculate_detailed_loss(criterion, mel_output, mel_target):
    """Detaylı kayıp hesaplaması yapar"""
    # Mel spektrogramları aynı boyuta getir
    max_len = min(mel_output.size(1), mel_target.size(1))
    mel_output = mel_output[:, :max_len, :]
    mel_target = mel_target[:, :max_len, :]
    
    mel_loss = criterion.mel_loss(mel_output, mel_target)
    duration_loss = criterion.duration_loss(mel_output, mel_target)
    pitch_loss = criterion.pitch_loss(mel_output, mel_target)
    energy_loss = criterion.energy_loss(mel_output, mel_target)
    
    total_loss = mel_loss + duration_loss + pitch_loss + energy_loss
    
    return {
        'total_loss': total_loss,
        'mel_loss': mel_loss.item(),
        'duration_loss': duration_loss.item(),
        'pitch_loss': pitch_loss.item(),
        'energy_loss': energy_loss.item()
    }

def plot_mel_comparison(mel_pred, mel_target):
    """Tahmin edilen ve hedef mel spektrogramlarını karşılaştırır"""
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    axes[0].imshow(mel_target.cpu().detach().numpy(), origin='lower', aspect='auto')
    axes[0].set_title('Orijinal Mel Spektrogram')
    axes[0].set_xlabel('Frame')
    axes[0].set_ylabel('Mel Filtre')
    
    axes[1].imshow(mel_pred.cpu().detach().numpy(), origin='lower', aspect='auto')
    axes[1].set_title('Üretilen Mel Spektrogram')
    axes[1].set_xlabel('Frame')
    axes[1].set_ylabel('Mel Filtre')
    
    plt.tight_layout()
    return fig

def validate(model, val_loader, criterion, device, writer, global_step):
    """Validation işlemi"""
    model.eval()
    total_val_loss = 0
    total_val_accuracy = 0
    
    # Validation örnekleri için
    mel_examples = []
    
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            # Batch verilerini al
            phonemes = batch['phonemes'].to(device)
            mel_target = batch['mel_spectrogram'].to(device)
            duration = batch['duration'].to(device)
            phoneme_length = batch['phoneme_length'].to(device)
            mel_length = batch['mel_length'].to(device)
            
            # Forward pass
            mel_output, _, _, _ = model(
                phonemes,
                phoneme_length,
                duration,
                mel_length
            )
            
            # Mel spektrogramları aynı boyuta getir
            max_len = min(mel_output.size(1), mel_target.size(1))
            mel_output = mel_output[:, :max_len, :]
            mel_target = mel_target[:, :max_len, :]
            
            # Loss hesapla
            losses = calculate_detailed_loss(criterion, mel_output, mel_target)
            total_val_loss += losses['total_loss'].item()
            
            # Accuracy hesapla
            accuracy = calculate_mel_accuracy(mel_output, mel_target)
            total_val_accuracy += accuracy
            
            # İlk 5 örneği sakla
            if i < 5:
                mel_examples.append((mel_output[0], mel_target[0]))
    
    # Ortalama loss ve accuracy hesapla
    avg_val_loss = total_val_loss / len(val_loader)
    avg_val_accuracy = total_val_accuracy / len(val_loader)
    
    # Tensorboard'a logla
    writer.add_scalar('Validation/Loss', avg_val_loss, global_step)
    writer.add_scalar('Validation/Accuracy', avg_val_accuracy, global_step)
    
    # Mel spektrogramları görselleştir
    for idx, (mel_pred, mel_target) in enumerate(mel_examples):
        fig = plot_mel_comparison(mel_pred, mel_target)
        writer.add_figure(f'Validation/Mel_Comparison_{idx}', fig, global_step)
        plt.close(fig)
    
    model.train()
    return avg_val_loss, avg_val_accuracy

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
    
    # Dataset'i eğitim ve validation olarak böl
    total_size = len(dataset)
    val_size = int(0.1 * total_size)  # %10 validation
    train_size = total_size - val_size
    
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=ModelConfig.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=dataset.collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=ModelConfig.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=dataset.collate_fn
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

    if args.resume:
        print(f"Checkpoint yükleniyor: {args.resume}")
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if checkpoint['scaler'] is not None:
            scaler.load_state_dict(checkpoint['scaler'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Eğitim epoch {start_epoch}'den devam ediyor")
    else:
        start_epoch = 1
    
    # Training loop
    global_step = 0
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    loss_window_size = 10
    loss_history = []
    
    for epoch in range(start_epoch, ModelConfig.epochs + 1):
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
                max_len = min(mel_output.size(1), mel_target.size(1))
                mel_output = mel_output[:, :max_len, :]
                mel_target = mel_target[:, :max_len, :]
                
                losses = calculate_detailed_loss(criterion, mel_output, mel_target)
                loss = losses['total_loss']
                
                # Accuracy hesapla
                accuracy = calculate_mel_accuracy(mel_output, mel_target)
                total_accuracy += accuracy
            
            # Backward pass
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            avg_loss = total_loss / (i + 1)
            avg_accuracy = total_accuracy / (i + 1)
            
            loss_history.append(loss.item())
            if len(loss_history) > loss_window_size:
                loss_history.pop(0)
            smoothed_loss = sum(loss_history) / len(loss_history)
            
            # Progress bar güncelle
            progress_bar.set_postfix({
                'Smoothed_Loss': f"{smoothed_loss:.4f}",
                'Loss': f"{losses['total_loss']:.4f}",
                'Mel': f"{losses['mel_loss']:.4f}",
                'Dur': f"{losses['duration_loss']:.4f}",
                'Pitch': f"{losses['pitch_loss']:.4f}",
                'Energy': f"{losses['energy_loss']:.4f}",
                'Acc': f"{avg_accuracy:.2f}%"
            })
            
            # Tensorboard'a logla
            step = (epoch - 1) * len(train_loader) + i
            writer.add_scalar('Loss/total', losses['total_loss'], step)
            writer.add_scalar('Loss/mel', losses['mel_loss'], step)
            writer.add_scalar('Loss/duration', losses['duration_loss'], step)
            writer.add_scalar('Loss/pitch', losses['pitch_loss'], step)
            writer.add_scalar('Loss/energy', losses['energy_loss'], step)
            writer.add_scalar('Accuracy/train', accuracy, step)
            writer.add_scalar('Learning_rate', scheduler.get_last_lr()[0], global_step)
            writer.add_scalar('Gradient_norm', torch.nn.utils.clip_grad_norm_(model.parameters(), -1), global_step)
            
            # Her 1000 adımda bir validation yap
            if global_step % 1000 == 0:
                val_loss, val_accuracy = validate(model, val_loader, criterion, device, writer, global_step)
                progress_bar.set_postfix({
                    'Loss': f"{losses['total_loss']:.4f}",
                    'Val Loss': f"{val_loss:.4f}",
                    'Val Acc': f"{val_accuracy:.2f}%"
                })
            
            global_step += 1
            
        
        # Epoch sonunda checkpoint kaydet
        if epoch % ModelConfig.save_step == 0:
            save_checkpoint(
                model, optimizer, scaler, epoch,
                os.path.join(args.checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
            )
            
        # Learning rate'i güncelle
        scheduler.step()
        
        # Her epoch sonunda validation
        val_loss, val_accuracy = validate(model, val_loader, criterion, device, writer, global_step)
        
        # Early stopping kontrolü
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # En iyi modeli kaydet
            save_checkpoint(model, optimizer, scaler, epoch,
                          os.path.join(args.checkpoint_dir, 'best_model.pt'))
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch} epochs")
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='FastSpeech2 Eğitim Scripti')
    parser.add_argument('--csv_path', type=str, default='C:\\Users\\sabri\\OneDrive\\Masaüstü\\TTSModel\\data\\preprocessed_train_with_phonemes.csv',
                        help='İşlenmiş CSV dosyasının yolu')
    parser.add_argument('--json_path', type=str, default='C:\\Users\\sabri\\OneDrive\\Masaüstü\\TTSModel\\data\\train_metadata.json',
                        help='Metadata JSON dosyasının yolu')
    parser.add_argument('--checkpoint_dir', type=str, default='C:\\Users\\sabri\\OneDrive\\Masaüstü\\TTSModel\\checkpoints',
                        help='Checkpoint kayıt dizini')
    parser.add_argument('--log_dir', type=str, default='C:\\Users\\sabri\\AppData\\Local\\Temp\\fast_speech_logs',
                        help='Tensorboard log dizini')
    parser.add_argument('--resume', type=str, default=None,
                        help='Eğitimi devam ettirmek için checkpoint dosyası')
    
    args = parser.parse_args()
    
    # Create directories if not exist
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    main(args)