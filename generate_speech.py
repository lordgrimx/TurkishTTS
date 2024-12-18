# generate_speech.py
import torch
from config.model_config import ModelConfig
from model.tts_pipeline import TTSPipeline
import subprocess
import re

class Dict2Obj:
    """Dictionary'yi object'e çeviren basit bir sınıf"""
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                setattr(self, key, Dict2Obj(value))
            else:
                setattr(self, key, value)

class EspeakPhonemizer:
    def __init__(self):
        # Fonem sözlüğünü modelin beklediği boyuta göre (65) düzenleyelim
        self.phoneme_dict = {
            # Temel sesli harfler (1-8)
            'a': 1, 'e': 2, 'ı': 3, 'i': 4, 'o': 5, 'ö': 6, 'u': 7, 'ü': 8,
            
            # Temel ünsüzler (9-29)
            'b': 9, 'c': 10, 'ç': 11, 'd': 12, 'f': 13, 'g': 14, 'ğ': 15,
            'h': 16, 'j': 17, 'k': 18, 'l': 19, 'm': 20, 'n': 21, 'p': 22,
            'r': 23, 's': 24, 'ş': 25, 't': 26, 'v': 27, 'y': 28, 'z': 29,
            
            # IPA karşılıkları (aynı ID'leri kullanıyor)
            'ɯ': 3,    # ı
            'ɛ': 2,    # e
            'œ': 6,    # ö
            'y': 8,    # ü
            'ɪ': 4,    # i
            'ʊ': 7,    # u
            'ø': 6,    # ö
            'ə': 2,    # e
            'ʧ': 11,   # ç
            'ʤ': 10,   # c
            'ɡ': 14,   # g
            'ɰ': 15,   # ğ
            'ʒ': 17,   # j
            'ʃ': 25,   # ş
            'ɾ': 23,   # r
            
            # Noktalama ve özel karakterler (30-35)
            ' ': 30,   # boşluk
            '.': 31,   # nokta
            ',': 32,   # virgül
            '?': 33,   # soru işareti
            '!': 34,   # ünlem
            '_': 35,   # sessizlik
            
            # Vurgu işaretleri (boşluk olarak ele alınıyor)
            'ˈ': 30,
            'ˌ': 30,
            'ː': 30,
        }
    
    def text_to_phonemes(self, text):
        try:
            # UTF-8 encoding ile çalıştır
            cmd = ['espeak-ng', '-q', '--ipa=3', '-v', 'tr', text]
            result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
            phonetic = result.stdout.strip()
            
            # Gereksiz karakterleri temizle
            phonetic = re.sub(r'[^\w\s\u0300-\u036f]', '', phonetic)
            phonetic = ''.join(char.lower() for char in phonetic if char in self.phoneme_dict)
            
            print(f"Temizlenmiş fonetik çıktı: {phonetic}")
            
            # Fonemleri ID'lere dönüştür
            phoneme_ids = [self.phoneme_dict[phone] for phone in phonetic if phone in self.phoneme_dict]
            
            if not phoneme_ids:
                raise ValueError("Fonem listesi boş!")
            
            return torch.LongTensor([phoneme_ids])
            
        except Exception as e:
            print(f"Fonem dönüşümü hatası: {str(e)}")
            return None

def main():
    # Config'i hazırla
    config = ModelConfig()
    # HifiGAN config'ini Dict2Obj'ye çevir
    config.hifigan_config = Dict2Obj(config.hifigan_config)
    
    # Pipeline'ı başlat
    pipeline = TTSPipeline(config)
    
    # GPU'ya taşı
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = pipeline.to(device)
    
    # Checkpoint'leri yükle
    pipeline.load_checkpoints(
        fastspeech2_path="checkpoints/checkpoint_epoch_20.pt",
        hifigan_path="hifi_gan/checkpoints/g_00010000"
    )
    
    # Fonemizer oluştur
    phonemizer = EspeakPhonemizer()
    
    # Test metinleri
    test_texts = [
        "Merhaba dünya",
        "Nasılsınız?",
        "Bu bir test cümlesidir."
    ]
    
    # Her metin için ses üret
    for i, text in enumerate(test_texts):
        print(f"\nMetin {i+1}: {text}")
        
        # Ses üret
        audio = pipeline.generate_speech(text, phonemizer)
        
        # Sesi kaydet
        if audio is not None:
            output_file = f"test_output_{i+1}.wav"
            pipeline.save_audio(audio, output_file)
            print(f"Ses dosyası oluşturuldu: {output_file}")

if __name__ == "__main__":
    main()