import os
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
matplotlib.interactive(True)
import matplotlib.pyplot as plt
import librosa
import librosa.display

def analyze_mel_spectrogram(mel_path, title="Mel Spectrogram Analysis"):
    """Mel spektrogramı analiz eder ve görselleştirir"""
    
    # Mel spektrogramı yükle
    mel = np.load(mel_path)
    
    # Temel istatistikler
    print("\nMel Spektrogram İstatistikleri:")
    print(f"Şekil: {mel.shape}")
    print(f"Minimum değer: {mel.min():.4f}")
    print(f"Maksimum değer: {mel.max():.4f}")
    print(f"Ortalama: {mel.mean():.4f}")
    print(f"Standart sapma: {mel.std():.4f}")
    
    # Görselleştirme
    plt.figure(figsize=(15, 10))
    
    # Mel spektrogram
    plt.subplot(2, 2, 1)
    librosa.display.specshow(mel, y_axis='mel', x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram')
    
    # Değer dağılımı (histogram)
    plt.subplot(2, 2, 2)
    plt.hist(mel.flatten(), bins=100)
    plt.title('Değer Dağılımı')
    plt.xlabel('Değer')
    plt.ylabel('Sıklık')
    
    # Zaman ekseni boyunca ortalama
    plt.subplot(2, 2, 3)
    plt.plot(mel.mean(axis=1))
    plt.title('Frekans Bandı Ortalamaları')
    plt.xlabel('Mel Bandı')
    plt.ylabel('Ortalama Değer')
    
    # Frekans ekseni boyunca ortalama
    plt.subplot(2, 2, 4)
    plt.plot(mel.mean(axis=0))
    plt.title('Zaman Ekseni Ortalaması')
    plt.xlabel('Zaman Çerçevesi')
    plt.ylabel('Ortalama Değer')
    
    plt.tight_layout()
    output_path = os.path.abspath(mel_path.replace('.npy', '_analysis.png'))
    plt.savefig(output_path)
    print(f"\nGrafik şuraya kaydedildi: {output_path}")
    plt.show()
    plt.close()
    
    # Ek kontroller
    print("\nKalite Kontrolleri:")
    
    # NaN veya sonsuz değer kontrolü
    has_nan = np.isnan(mel).any()
    has_inf = np.isinf(mel).any()
    print(f"NaN değer var mı?: {has_nan}")
    print(f"Sonsuz değer var mı?: {has_inf}")
    
    # Değer aralığı kontrolü (tipik mel spektrogramlar için)
    normal_range = (-20, 0)  # dB cinsinden tipik aralık
    values_in_range = np.logical_and(mel >= normal_range[0], mel <= normal_range[1])
    percent_in_range = (values_in_range.sum() / mel.size) * 100
    print(f"Değerlerin {percent_in_range:.2f}%'i normal aralıkta ({normal_range[0]} ile {normal_range[1]} dB arası)")
    
    # Sessizlik/aktivite analizi
    activity_threshold = -10  # dB
    active_frames = (mel.mean(axis=0) > activity_threshold).sum()
    total_frames = mel.shape[1]
    activity_ratio = (active_frames / total_frames) * 100
    print(f"Aktif çerçeve oranı: {activity_ratio:.2f}%")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mel_path', required=True, help='Analiz edilecek mel spektrogram dosyası (.npy)')
    args = parser.parse_args()
    
    analyze_mel_spectrogram(args.mel_path) 