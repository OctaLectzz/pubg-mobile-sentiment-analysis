"""
Naïve Bayes Classifier dengan N-Gram untuk Analisis Sentimen.
"""
import os, pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

# N-Gram configurations
NGRAM_OPTIONS = {
    "Unigram (1,1)": (1, 1),
    "Bigram (2,2)": (2, 2),
    "Trigram (3,3)": (3, 3),
    "Unigram+Bigram (1,2)": (1, 2),
    "Unigram+Bigram+Trigram (1,3)": (1, 3),
}


def build_model(texts, labels, ngram_range=(1, 2), test_size=0.2, random_state=42):
    """
    Build dan train Naïve Bayes model dengan TF-IDF N-Gram.
    
    Returns:
        dict: model, vectorizer, metrics, predictions, dll.
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    
    # TF-IDF Vectorizer dengan N-Gram
    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range,
        max_features=10000,
        min_df=1,
        max_df=0.95,
        sublinear_tf=True
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Train Multinomial Naïve Bayes
    model = MultinomialNB(alpha=1.0)
    model.fit(X_train_tfidf, y_train)
    
    # Predictions
    y_pred = model.predict(X_test_tfidf)
    y_pred_proba = model.predict_proba(X_test_tfidf)
    
    # Metrics
    labels_unique = sorted(labels.unique())
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=labels_unique)
    report = classification_report(y_test, y_pred, labels=labels_unique, zero_division=0, output_dict=True)
    
    return {
        "model": model,
        "vectorizer": vectorizer,
        "ngram_range": ngram_range,
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "y_pred": y_pred, "y_pred_proba": y_pred_proba,
        "accuracy": acc, "precision": prec, "recall": rec, "f1_score": f1,
        "confusion_matrix": cm, "classification_report": report,
        "labels": labels_unique,
        "train_size": len(X_train), "test_size": len(X_test),
        "feature_names": vectorizer.get_feature_names_out(),
    }


def predict_text(text, model, vectorizer):
    """Prediksi sentimen untuk satu teks."""
    text_tfidf = vectorizer.transform([text])
    pred = model.predict(text_tfidf)[0]
    proba = model.predict_proba(text_tfidf)[0]
    classes = model.classes_
    proba_dict = {c: round(float(p)*100, 2) for c, p in zip(classes, proba)}
    return pred, proba_dict


def compare_ngrams(texts, labels, test_size=0.2, random_state=42):
    """Bandingkan performa berbagai konfigurasi N-Gram."""
    results = []
    for name, ngram in NGRAM_OPTIONS.items():
        res = build_model(texts, labels, ngram_range=ngram, test_size=test_size, random_state=random_state)
        results.append({
            "N-Gram": name,
            "Range": str(ngram),
            "Accuracy": round(res["accuracy"]*100, 2),
            "Precision": round(res["precision"]*100, 2),
            "Recall": round(res["recall"]*100, 2),
            "F1-Score": round(res["f1_score"]*100, 2),
            "Features": len(res["feature_names"]),
        })
    return pd.DataFrame(results)


def save_model(model, vectorizer, model_dir=None):
    """Simpan model dan vectorizer ke file pickle."""
    if model_dir is None: model_dir = MODEL_DIR
    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, "model.pkl"), "wb") as f:
        pickle.dump(model, f)
    with open(os.path.join(model_dir, "vectorizer.pkl"), "wb") as f:
        pickle.dump(vectorizer, f)
    print(f"[INFO] Model disimpan ke {model_dir}")


def load_model(model_dir=None):
    """Load model dan vectorizer dari file pickle."""
    if model_dir is None: model_dir = MODEL_DIR
    with open(os.path.join(model_dir, "model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(model_dir, "vectorizer.pkl"), "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer
