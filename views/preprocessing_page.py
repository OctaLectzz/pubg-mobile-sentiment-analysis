"""Preprocessing Page — text cleaning pipeline and before/after comparison."""
import streamlit as st
from views import redirect_to_train


def render(t):
    """Render the Preprocessing page."""
    st.markdown(f"### 🔧 {t.get('nav_preprocessing', 'Preprocessing')}")
    st.markdown(f"_{t.get('preprocess_page_desc', 'Pipeline pembersihan teks untuk mempersiapkan data sebelum klasifikasi.')}_")

    # ── Pipeline Steps ──
    st.markdown(f"##### 🔄 {t.get('pipeline_title', 'Pipeline Preprocessing')}")

    steps = [
        ("1", "Case Folding", t.get("step_casefolding", "Mengubah semua teks menjadi huruf kecil (lowercase).")),
        ("2", "Normalisasi Karakter", t.get("step_normalize", "Menormalkan karakter berulang: 'baguuuus' → 'bagus'.")),
        ("3", "Pembersihan Teks", t.get("step_cleaning", "Menghapus URL, mention, hashtag, emoji, angka, dan karakter spesial.")),
        ("4", "Normalisasi Slang", t.get("step_slang", "Mengubah kata slang/singkatan ke bentuk baku menggunakan kamus.")),
        ("5", "Penanganan Negasi", t.get("step_negation", "Menggabungkan kata negasi dengan kata berikutnya: 'tidak bagus' → 'tidak_bagus'.")),
        ("6", "Stopword Removal", t.get("step_stopword", "Menghapus kata-kata umum yang tidak bermakna, dengan tetap menyimpan kata sentimen.")),
        ("7", "Stemming", t.get("step_stemming", "Mengubah kata ke bentuk dasar menggunakan PySastrawi (opsional).")),
    ]

    cols = st.columns(4)
    for i, (num, title, desc) in enumerate(steps):
        with cols[i % 4]:
            st.markdown(f"""
            <div class="glass-card" style="padding:18px; margin-bottom:12px; min-height:130px;">
                <div style="display:flex; align-items:center; gap:10px; margin-bottom:8px;">
                    <span class="step-badge">{num}</span>
                    <strong style="font-size:.88rem;">{title}</strong>
                </div>
                <p style="opacity:.7; font-size:.82rem; margin:0; line-height:1.4;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)

    stemming_status = "✅ Aktif" if st.session_state.get("use_stemming", True) else "❌ Nonaktif"
    st.info(f"🔧 **Stemming:** {stemming_status} — " + t.get("stemming_note", "Ubah di konfigurasi model pada sidebar."))

    st.markdown("---")

    # ── Before/After Comparison ──
    df_proc = st.session_state.get("df_processed")
    if df_proc is None:
        redirect_to_train(t)
        return

    st.markdown(f"##### 📝 {t['before_after']}")

    # Stats row
    total_words_before = df_proc["review_text"].apply(lambda x: len(str(x).split())).sum()
    total_words_after = df_proc["cleaned_text"].apply(lambda x: len(str(x).split())).sum()
    vocab_before = len(set(" ".join(df_proc["review_text"].astype(str)).split()))
    vocab_after = len(set(" ".join(df_proc["cleaned_text"]).split()))

    s1, s2, s3, s4 = st.columns(4)
    s1.metric(t.get("words_before", "Total Kata (Sebelum)"), f"{total_words_before:,}")
    s2.metric(t.get("words_after", "Total Kata (Sesudah)"), f"{total_words_after:,}")
    s3.metric(t.get("vocab_before", "Vocabulary (Sebelum)"), f"{vocab_before:,}")
    s4.metric(t.get("vocab_after", "Vocabulary (Sesudah)"), f"{vocab_after:,}")

    st.markdown("---")

    # Comparison table
    compare_df = df_proc[["review_text", "cleaned_text", "sentiment"]].head(20).copy()
    compare_df.columns = [t["original_review"], t["after_preprocess"], t["sentiment"]]
    st.dataframe(compare_df, use_container_width=True, height=400)
