import numpy as np
import soundfile as sf

def npy_to_wav(npy_file, wav_file, sample_rate=44100):
    """
    .npy dosyasını .wav dosyasına dönüştürür.
    
    Args:
        npy_file (str): Yüklenecek .npy dosyasının yolu.
        wav_file (str): Kaydedilecek .wav dosyasının yolu.
        sample_rate (int): Ses dosyasının örnekleme hızı (varsayılan: 44100).
    """
    try:
        # .npy dosyasını yükle
        audio_data = np.load(npy_file)
        
        # Verilerin normalize edilmesi
        if np.max(np.abs(audio_data)) > 1:
            audio_data = audio_data / np.max(np.abs(audio_data))
        
        # .wav dosyasına yaz
        sf.write(wav_file, audio_data, sample_rate)
        print(f"{wav_file} başarıyla oluşturuldu.")
    except Exception as e:
        print(f"Hata: {e}")

# Örnek kullanım
npy_file = "test_output_mel.npy"  # .npy dosyasının yolu
wav_file = "output.wav"   # .wav dosyasının kaydedileceği yol
sample_rate = 44100       # Örnekleme hızı (Hz)

npy_to_wav(npy_file, wav_file, sample_rate)
