"""
Text Preprocessing Pipeline untuk Analisis Sentimen Bahasa Indonesia.
Includes negation handling, character normalization, and configurable stopwords.
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

# Sentiment-bearing words that should NOT be removed as stopwords
# These are crucial for accurate sentiment classification
KEEP_WORDS = {
    "tidak", "bukan", "jangan", "belum", "tanpa", "kurang",
    "sangat", "sekali", "banget", "paling", "lebih", "terlalu",
    "bagus", "jelek", "buruk", "baik", "seru", "bosan",
    "suka", "benci", "senang", "kecewa", "puas", "marah",
}

# Negation words for bigram handling
NEGATION_WORDS = {"tidak", "bukan", "jangan", "belum", "tanpa", "kurang", "tak", "tiada"}

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


def normalize_repeated_chars(text):
    """
    Normalize repeated characters: 'baguuuus' -> 'bagus', 'seruuuu' -> 'seru'.
    Keeps max 2 consecutive identical characters.
    """
    return re.sub(r"(.)\1{2,}", r"\1\1", text)


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


def handle_negation(text):
    """
    Handle negation by combining negation word with the next word.
    'tidak bagus' -> 'tidak_bagus'
    This preserves negation context for the classifier.
    """
    words = text.split()
    result = []
    i = 0
    while i < len(words):
        if words[i] in NEGATION_WORDS and i + 1 < len(words):
            result.append(f"{words[i]}_{words[i + 1]}")
            i += 2
        else:
            result.append(words[i])
            i += 1
    return " ".join(result)


def remove_stopwords(text):
    """
    Hapus stopwords bahasa Indonesia.
    Keeps sentiment-bearing words that are important for classification.
    """
    words = text.split()
    filtered = [
        w for w in words
        if (w not in STOPWORDS or w in KEEP_WORDS or "_" in w) and len(w) > 1
    ]
    return " ".join(filtered)


def stem_text(text):
    """Stemming menggunakan Sastrawi. Skips negation bigrams."""
    stemmer = get_stemmer()
    words = text.split()
    stemmed = []
    for w in words:
        if "_" in w:
            # Negation bigram: stem each part separately
            parts = w.split("_")
            stemmed.append("_".join(stemmer.stem(p) for p in parts))
        else:
            stemmed.append(stemmer.stem(w))
    return " ".join(stemmed)


def preprocess(text, use_stemming=True):
    """
    Pipeline preprocessing lengkap.
    1. Case folding
    2. Repeated char normalization
    3. Cleaning (URL, emoji, etc.)
    4. Normalisasi slang
    5. Negation handling
    6. Stopword removal (keeps sentiment words)
    7. Stemming (opsional)
    """
    text = case_folding(text)
    text = normalize_repeated_chars(text)
    text = clean_text(text)
    text = normalize_slang(text)
    text = handle_negation(text)
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
        "Game ini tidak bagus, sangat mengecewakan!",
        "Baguuuuus banget gamenya seruuuu!",
    ]
    for t in test_texts:
        print(f"Original : {t}")
        print(f"Processed: {preprocess(t)}")
        print()
