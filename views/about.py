"""About / Research Methodology Page."""
import streamlit as st


def render(t):
    """Render the About / Research page."""
    st.markdown(f"### 📖 {t.get('nav_about', 'Tentang Penelitian')}")

    # ── Research Overview ──
    st.markdown(f"""
    <div class="hero-banner" style="padding:28px 32px;">
        <h2 style="margin:0 0 8px;">📚 {t.get('about_title', 'Analisis Sentimen Review PUBG Mobile')}</h2>
        <p style="margin:0; opacity:.9;">{t.get('about_subtitle', 'Menggunakan Metode Naïve Bayes Classifier dengan Ekstraksi Fitur N-Gram')}</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Methodology ──
    st.markdown(f"##### 🔬 {t.get('about_methodology', 'Metodologi Penelitian')}")

    steps = [
        ("1️⃣", t.get("method_collect", "Pengumpulan Data"),
         t.get("method_collect_desc",
                "Data review dikumpulkan dari Apple App Store menggunakan RSS Feed API. "
                "Review dari berbagai negara dapat diambil untuk analisis komparatif.")),
        ("2️⃣", t.get("method_label", "Pelabelan Sentimen"),
         t.get("method_label_desc",
                "Sentimen ditentukan berdasarkan rating: "
                "Rating 1-2 → Negatif, Rating 3 → Netral, Rating 4-5 → Positif.")),
        ("3️⃣", t.get("method_preprocess", "Preprocessing Teks"),
         t.get("method_preprocess_desc",
                "Pipeline NLP khusus Bahasa Indonesia meliputi case folding, "
                "normalisasi karakter berulang, pembersihan teks, normalisasi slang, "
                "penanganan negasi, stopword removal, dan stemming menggunakan PySastrawi.")),
        ("4️⃣", t.get("method_feature", "Ekstraksi Fitur"),
         t.get("method_feature_desc",
                "Fitur diekstrak menggunakan TF-IDF Vectorizer dengan konfigurasi N-Gram yang dapat disesuaikan "
                "(Unigram, Bigram, Trigram, atau kombinasi).")),
        ("5️⃣", t.get("method_classify", "Klasifikasi"),
         t.get("method_classify_desc",
                "Complement Naïve Bayes Classifier digunakan untuk klasifikasi sentimen. "
                "GridSearchCV dipakai untuk menemukan hyperparameter alpha optimal.")),
    ]

    for num, title, desc in steps:
        st.markdown(f"""
        <div class="glass-card" style="padding:18px 22px; margin-bottom:10px;">
            <div style="display:flex; align-items:flex-start; gap:14px;">
                <span style="font-size:1.5rem;">{num}</span>
                <div>
                    <strong style="font-size:.95rem;">{title}</strong>
                    <p style="opacity:.7; font-size:.85rem; margin:6px 0 0; line-height:1.5;">{desc}</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Naïve Bayes Explanation ──
    st.markdown(f"##### 🧮 {t.get('about_nb', 'Algoritma Naïve Bayes')}")

    st.markdown(f"""
    <div class="glass-card" style="padding:22px;">
        <p style="line-height:1.7; margin:0 0 14px;">
            {t.get('nb_desc', 'Naïve Bayes adalah algoritma klasifikasi probabilistik berdasarkan Teorema Bayes dengan asumsi independensi antar fitur. Dalam konteks analisis sentimen, setiap kata (atau N-Gram) dalam review dianggap sebagai fitur independen yang berkontribusi terhadap probabilitas kelas sentimen.')}
        </p>
        <div style="background:rgba(102,126,234,.1); border-radius:10px; padding:16px; text-align:center; margin:10px 0;">
            <p style="font-size:1.1rem; font-weight:600; margin:0; font-family:monospace;">
                P(C|X) = P(X|C) × P(C) / P(X)
            </p>
        </div>
        <p style="opacity:.7; font-size:.85rem; margin:10px 0 0; line-height:1.5;">
            {t.get('nb_formula_desc', 'Di mana P(C|X) adalah probabilitas kelas C diberikan fitur X, P(X|C) adalah likelihood, P(C) adalah prior probability, dan P(X) adalah evidence.')}
        </p>
        <p style="opacity:.7; font-size:.85rem; margin:10px 0 0; line-height:1.5;">
            {t.get('nb_complement_desc', '<strong>Complement Naïve Bayes</strong> adalah varian yang menghitung probabilitas menggunakan data dari kelas komplemen, sehingga lebih efektif untuk dataset yang tidak seimbang (imbalanced).')}
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── N-Gram Explanation ──
    st.markdown(f"##### 📐 {t.get('about_ngram', 'Fitur N-Gram')}")

    st.markdown(t.get("ngram_explain", """
    N-Gram adalah urutan N item dari sampel teks. Dalam analisis sentimen, N-Gram digunakan untuk menangkap konteks kata
    yang tidak bisa ditangkap oleh fitur kata tunggal (Unigram).

    | Tipe | Range | Contoh |
    |---|---|---|
    | Unigram | (1,1) | "game", "bagus", "lag" |
    | Bigram | (2,2) | "game bagus", "sangat seru" |
    | Trigram | (3,3) | "game sangat bagus" |
    | Unigram+Bigram | (1,2) | Kombinasi keduanya |
    | Unigram+Bigram+Trigram | (1,3) | Semua kombinasi |
    """))

    st.markdown("---")

    # ── Tech Stack ──
    st.markdown(f"##### 🛠️ {t.get('about_tech', 'Teknologi yang Digunakan')}")

    tech_items = [
        ("🐍", "Python 3.x", t.get("tech_python", "Bahasa pemrograman utama")),
        ("🌐", "Streamlit", t.get("tech_streamlit", "Framework web untuk dashboard interaktif")),
        ("🤖", "Scikit-Learn", t.get("tech_sklearn", "Machine learning (Naïve Bayes, TF-IDF)")),
        ("📝", "PySastrawi", t.get("tech_sastrawi", "Stemmer Bahasa Indonesia")),
        ("📊", "Plotly", t.get("tech_plotly", "Visualisasi data interaktif")),
        ("☁️", "WordCloud", t.get("tech_wordcloud", "Generasi word cloud")),
        ("🐼", "Pandas & NumPy", t.get("tech_pandas", "Manipulasi dan analisis data")),
    ]

    tech_cols = st.columns(4)
    for i, (icon, name, desc) in enumerate(tech_items):
        with tech_cols[i % 4]:
            st.markdown(f"""
            <div class="glass-card" style="padding:16px; text-align:center; margin-bottom:10px; min-height:110px;">
                <div style="font-size:1.8rem; margin-bottom:8px;">{icon}</div>
                <strong style="font-size:.85rem;">{name}</strong>
                <p style="opacity:.6; font-size:.78rem; margin:4px 0 0;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Credits ──
    st.markdown(f"##### 👨‍💻 {t.get('about_credits', 'Kredit')}")
    st.markdown(f"""
    <div class="glass-card" style="padding:22px; text-align:center;">
        <p style="font-size:1rem; margin:0 0 6px; font-weight:600;">PUBG Mobile Sentiment Analysis Dashboard</p>
        <p style="opacity:.7; font-size:.88rem; margin:0 0 4px;">{t.get('about_method_label', 'Metode: Naïve Bayes Classifier dengan N-Gram Feature Extraction')}</p>
        <p style="opacity:.5; font-size:.82rem; margin:12px 0 0;">© 2025 — OctaLectzz</p>
    </div>
    """, unsafe_allow_html=True)
