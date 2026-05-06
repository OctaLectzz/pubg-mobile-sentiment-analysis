"""
Text Preprocessing Pipeline untuk Analisis Sentimen Bahasa Indonesia.
"""
import os, re, json
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load slang dictionary
SLANG_PATH = os.path.join(BASE_DIR, "data", "slang_dict.json")
if os.path.exists(SLANG_PATH):
    with open(SLANG_PATH, "r", encoding="utf-8") as f:
        SLANG_DICT = json.load(f)
else:
    SLANG_DICT = {}

# Load stopwords
STOPWORD_PATH = os.path.join(BASE_DIR, "data", "stopwords_id.txt")
if os.path.exists(STOPWORD_PATH):
    with open(STOPWORD_PATH, "r", encoding="utf-8") as f:
        STOPWORDS = set(f.read().strip().split("\n"))
else:
    STOPWORDS = set()

# Stemmer (singleton)
_stemmer = None
def get_stemmer():
    global _stemmer
    if _stemmer is None:
        factory = StemmerFactory()
        _stemmer = factory.create_stemmer()
    return _stemmer


def case_folding(text):
    """Ubah ke lowercase."""
    return str(text).lower().strip()


def clean_text(text):
    """Hapus URL, mention, hashtag, emoji, angka, karakter spesial."""
    text = re.sub(r"http\S+|www\.\S+", "", text)  # URL
    text = re.sub(r"@\w+", "", text)  # mention
    text = re.sub(r"#\w+", "", text)  # hashtag
    # Hapus emoji
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF"
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE
    )
    text = emoji_pattern.sub("", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)  # non-alpha
    text = re.sub(r"\s+", " ", text).strip()  # multi-space
    return text


def normalize_slang(text):
    """Normalisasi kata slang menggunakan kamus."""
    words = text.split()
    normalized = [SLANG_DICT.get(w, w) for w in words]
    return " ".join(normalized)


def remove_stopwords(text):
    """Hapus stopwords bahasa Indonesia."""
    words = text.split()
    filtered = [w for w in words if w not in STOPWORDS and len(w) > 1]
    return " ".join(filtered)


def stem_text(text):
    """Stemming menggunakan Sastrawi."""
    stemmer = get_stemmer()
    return stemmer.stem(text)


def preprocess(text, use_stemming=True):
    """
    Pipeline preprocessing lengkap.
    1. Case folding
    2. Cleaning
    3. Normalisasi slang
    4. Stopword removal
    5. Stemming (opsional)
    """
    text = case_folding(text)
    text = clean_text(text)
    text = normalize_slang(text)
    text = remove_stopwords(text)
    if use_stemming:
        text = stem_text(text)
    return text


def preprocess_dataframe(df, text_column="review_text", use_stemming=True):
    """
    Preprocessing seluruh DataFrame.
    Menambahkan kolom 'cleaned_text' ke DataFrame.
    """
    import pandas as pd
    df = df.copy()
    total = len(df)
    cleaned = []
    for i, text in enumerate(df[text_column]):
        cleaned.append(preprocess(str(text), use_stemming=use_stemming))
        if (i + 1) % 50 == 0 or (i + 1) == total:
            print(f"  Preprocessing: {i+1}/{total}", end="\r")
    print()
    df["cleaned_text"] = cleaned
    # Hapus baris kosong setelah preprocessing
    df = df[df["cleaned_text"].str.strip().astype(bool)]
    return df


if __name__ == "__main__":
    # Test preprocessing
    test_texts = [
        "Game ini bagus bgt!! Grafisnya keren 😍😍 #PUBG @pubgmobile",
        "Gak bisa login udh 3 hari, server down terus 😡",
        "Lumayan lah buat ngisi waktu, tp agak lag di HP kentang",
    ]
    for t in test_texts:
        print(f"Original : {t}")
        print(f"Processed: {preprocess(t)}")
        print()
