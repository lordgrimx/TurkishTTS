def get_phoneme_to_id():
    """
    Fonem-ID eşleştirme sözlüğünü döndürür.
    """
    # Temel fonemler
    phonemes = [
        '_',  # Padding ve boşluk için
        'a', 'e', 'I', 'i', 'o', 'O', 'u', 'U',  # Ünlüler
        'b', 'd', 'f', 'g', 'G', 'h', 'j', 'k', 'l', 'm', 'n',  # Ünsüzler
        'p', 'r', 's', 'S', 't', 'v', 'z',  # Ünsüzler (devam)
        'dZ', 'tS',  # Bileşik sesler
        'Z'  # Ek sesler
    ]
    
    # Fonem-ID sözlüğünü oluştur
    return {phoneme: idx for idx, phoneme in enumerate(phonemes)}

def get_id_to_phoneme():
    """
    ID-Fonem eşleştirme sözlüğünü döndürür.
    """
    phoneme_to_id = get_phoneme_to_id()
    return {idx: phoneme for phoneme, idx in phoneme_to_id.items()}

def text_to_phonemes(text):
    """
    Türkçe metni fonem dizisine çevirir.
    Dataset'teki fonem setine uygun şekilde dönüşüm yapar.
    """
    # Metni küçük harfe çevir ve normalize et
    text = text.lower().strip()
    phonemes = []
    
    # Türkçe harf-fonem eşleştirmesi
    turkish_phonemes = {
        'a': 'a', 'e': 'e', 'ı': 'I', 'i': 'i', 'o': 'o', 'ö': 'O', 'u': 'u', 'ü': 'U',
        'b': 'b', 'c': 'dZ', 'ç': 'tS', 'd': 'd', 'f': 'f', 'g': 'g', 'ğ': 'G',
        'h': 'h', 'j': 'Z', 'k': 'k', 'l': 'l', 'm': 'm', 'n': 'n', 'p': 'p',
        'r': 'r', 's': 's', 'ş': 'S', 't': 't', 'v': 'v', 'y': 'j', 'z': 'z',
        ' ': '_'  # Boşluk için özel fonem
    }
    
    print(f"\nDönüştürülecek metin: {text}")
    
    # Her karakteri fonemlere çevir
    for char in text:
        if char in turkish_phonemes:
            phonemes.append(turkish_phonemes[char])
        else:
            print(f"Uyarı: '{char}' karakteri için fonem bulunamadı, atlanıyor")
    
    print(f"Oluşturulan fonemler: {phonemes}")
    return phonemes

def phonemes_to_ids(phonemes):
    """
    Fonem dizisini ID dizisine çevirir.
    """
    phoneme_to_id = get_phoneme_to_id()
    ids = []
    
    for phoneme in phonemes:
        if phoneme in phoneme_to_id:
            ids.append(phoneme_to_id[phoneme])
        else:
            print(f"Uyarı: '{phoneme}' fonemi için ID bulunamadı")
            # Bilinmeyen fonemler için özel bir ID kullanılabilir
            # ids.append(phoneme_to_id['_'])  # Varsayılan olarak boşluk ID'sini kullan
    
    return ids

def ids_to_phonemes(ids):
    """
    ID dizisini fonem dizisine çevirir.
    """
    id_to_phoneme = get_id_to_phoneme()
    phonemes = []
    
    for id in ids:
        if id in id_to_phoneme:
            phonemes.append(id_to_phoneme[id])
        else:
            print(f"Uyarı: {id} ID'si için fonem bulunamadı")
    
    return phonemes

def process_text(text, return_ids=True):
    """
    Metni işleyerek fonem ve/veya ID dizisine çevirir.
    
    Args:
        text (str): İşlenecek metin
        return_ids (bool): True ise ID dizisi, False ise fonem dizisi döndürür
        
    Returns:
        list: Fonem veya ID dizisi
    """
    phonemes = text_to_phonemes(text)
    if return_ids:
        ids = phonemes_to_ids(phonemes)
        print(f"Oluşturulan ID'ler: {ids}")
        return ids
    return phonemes

# Kullanım örneği:
if __name__ == "__main__":
    test_text = "merhaba dünya"
    
    # Sadece fonemlere çevir
    phonemes = process_text(test_text, return_ids=False)
    print(f"Fonemler: {phonemes}")
    
    # Fonemleri ID'lere çevir
    ids = process_text(test_text, return_ids=True)
    print(f"ID'ler: {ids}")
    
    # ID'leri tekrar fonemlere çevir (doğrulama için)
    back_to_phonemes = ids_to_phonemes(ids)
    print(f"ID'lerden geri çevrilen fonemler: {back_to_phonemes}")