def text_to_phonemes(text):
    """
    Türkçe metni fonem dizisine çevirir.
    Dataset'teki fonem setine uygun şekilde dönüşüm yapar.
    """
    # Metni küçük harfe çevir ve normalize et
    text = text.lower().strip()
    phonemes = []
    
    # Türkçe harf-fonem eşleştirmesi (dataset'e göre güncellenmeli)
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