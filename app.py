"""
Streamlit Dashboard — Analisis Sentimen Ulasan PUBG Mobile
Naïve Bayes Classifier dengan N-Gram Feature Extraction
"""
import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns

from scraper import generate_sample_data, OUTPUT_FILE
from preprocessing import preprocess, preprocess_dataframe
from model import (
    build_model, predict_text, compare_ngrams, save_model,
    NGRAM_OPTIONS,
)

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Analisis Sentimen PUBG Mobile",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── global ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── metric cards ── */
div[data-testid="stMetric"] {
    background: linear-gradient(135deg, #1e1e2f 0%, #2d2d44 100%);
    border: 1px solid rgba(255,255,255,.08);
    border-radius: 14px;
    padding: 18px 22px;
    box-shadow: 0 4px 24px rgba(0,0,0,.25);
}
div[data-testid="stMetric"] label { color: #a0a0c0 !important; font-size: .82rem; }
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    font-size: 1.8rem !important; font-weight: 700; color: #e0e0ff !important;
}

/* ── tabs ── */
button[data-baseweb="tab"] {
    font-weight: 600 !important; font-size: .95rem !important;
    border-radius: 10px 10px 0 0 !important;
}

/* ── sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #12121f 0%, #1a1a2e 100%);
}

/* ── tables ── */
.stDataFrame { border-radius: 12px; overflow: hidden; }

/* hero banner */
.hero-banner {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
    border-radius: 16px;
    padding: 32px 36px;
    margin-bottom: 24px;
    color: white;
    box-shadow: 0 8px 32px rgba(102,126,234,.35);
}
.hero-banner h1 { margin: 0 0 6px 0; font-size: 2rem; font-weight: 800; }
.hero-banner p  { margin: 0; opacity: .88; font-size: 1.05rem; }

/* sentiment badges */
.badge { display:inline-block; padding:4px 14px; border-radius:20px; font-weight:600; font-size:.82rem; }
.badge-positif  { background:#10b981; color:#fff; }
.badge-netral   { background:#f59e0b; color:#fff; }
.badge-negatif  { background:#ef4444; color:#fff; }
</style>
""", unsafe_allow_html=True)

# ── Colour palette ───────────────────────────────────────────────────────────
SENTIMENT_COLORS = {"Positif": "#10b981", "Netral": "#f59e0b", "Negatif": "#ef4444"}

# ── Session state helpers ────────────────────────────────────────────────────
if "df_raw" not in st.session_state:
    st.session_state.df_raw = None
if "df_processed" not in st.session_state:
    st.session_state.df_processed = None
if "model_result" not in st.session_state:
    st.session_state.model_result = None
if "ngram_comparison" not in st.session_state:
    st.session_state.ngram_comparison = None


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🎮 PUBG Mobile")
    st.markdown("### Analisis Sentimen")
    st.markdown("---")

    # ── Data source ──
    st.markdown("#### 📂 Sumber Data")
    data_source = st.radio(
        "Pilih sumber data",
        ["Data Default (Sample)", "Upload CSV"],
        label_visibility="collapsed",
    )

    if data_source == "Upload CSV":
        uploaded = st.file_uploader("Upload file CSV", type=["csv"])
        if uploaded is not None:
            st.session_state.df_raw = pd.read_csv(uploaded)
            st.success(f"✅ {len(st.session_state.df_raw)} baris dimuat")
    else:
        if st.button("🔄 Muat Data Sample", use_container_width=True):
            with st.spinner("Membuat data sample..."):
                generate_sample_data()
            st.session_state.df_raw = pd.read_csv(OUTPUT_FILE)
            st.success(f"✅ {len(st.session_state.df_raw)} review dimuat")

        # Auto-load if file exists and df_raw is empty
        if st.session_state.df_raw is None and os.path.exists(OUTPUT_FILE):
            st.session_state.df_raw = pd.read_csv(OUTPUT_FILE)

    st.markdown("---")

    # ── Version filter ──
    if st.session_state.df_raw is not None and "version" in st.session_state.df_raw.columns:
        versions = sorted(st.session_state.df_raw["version"].dropna().unique().tolist())
        selected_versions = st.multiselect(
            "🔖 Filter Versi Aplikasi",
            options=versions,
            default=versions,
        )
    else:
        selected_versions = None

    st.markdown("---")

    # ── N-Gram selector ──
    st.markdown("#### ⚙️ Konfigurasi Model")
    ngram_choice = st.selectbox("Tipe N-Gram", list(NGRAM_OPTIONS.keys()), index=3)
    use_stemming = st.checkbox("Gunakan Stemming", value=True)
    test_size = st.slider("Ukuran Test Set (%)", 10, 40, 20, 5) / 100

    st.markdown("---")

    # ── Train button ──
    btn_train = st.button("🚀 Latih Model", use_container_width=True, type="primary")
    btn_compare = st.button("📊 Bandingkan N-Gram", use_container_width=True)

    st.markdown("---")
    st.caption("© 2025 — Naïve Bayes N-Gram Sentiment Analysis")


# ══════════════════════════════════════════════════════════════════════════════
#  APPLY VERSION FILTER
# ══════════════════════════════════════════════════════════════════════════════
df_display = st.session_state.df_raw
if df_display is not None and selected_versions is not None:
    df_display = df_display[df_display["version"].isin(selected_versions)]


# ══════════════════════════════════════════════════════════════════════════════
#  PREPROCESSING & TRAINING LOGIC
# ══════════════════════════════════════════════════════════════════════════════
if btn_train and df_display is not None and len(df_display) > 0:
    with st.spinner("⏳ Preprocessing teks..."):
        df_proc = preprocess_dataframe(df_display, use_stemming=use_stemming)
        st.session_state.df_processed = df_proc

    with st.spinner("🧠 Melatih model Naïve Bayes..."):
        ngram_range = NGRAM_OPTIONS[ngram_choice]
        result = build_model(
            df_proc["cleaned_text"], df_proc["sentiment"],
            ngram_range=ngram_range, test_size=test_size,
        )
        st.session_state.model_result = result
        save_model(result["model"], result["vectorizer"])

    st.toast("✅ Model berhasil dilatih!", icon="🎉")

if btn_compare and df_display is not None and len(df_display) > 0:
    with st.spinner("⏳ Preprocessing teks..."):
        df_proc = preprocess_dataframe(df_display, use_stemming=use_stemming)
        st.session_state.df_processed = df_proc

    with st.spinner("📊 Membandingkan semua konfigurasi N-Gram..."):
        comparison = compare_ngrams(df_proc["cleaned_text"], df_proc["sentiment"], test_size=test_size)
        st.session_state.ngram_comparison = comparison

    st.toast("✅ Perbandingan selesai!", icon="📊")


# ══════════════════════════════════════════════════════════════════════════════
#  HERO BANNER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero-banner">
    <h1>🎮 Analisis Sentimen PUBG Mobile</h1>
    <p>Dashboard interaktif untuk menganalisis sentimen ulasan App Store menggunakan Naïve Bayes Classifier &amp; N-Gram</p>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Eksplorasi Data",
    "🔧 Preprocessing",
    "🧠 Model & Evaluasi",
    "🔮 Prediksi",
])


# ──────────────────────────────────────────────────────────────────────────────
#  TAB 1 — EKSPLORASI DATA
# ──────────────────────────────────────────────────────────────────────────────
with tab1:
    if df_display is None or len(df_display) == 0:
        st.info("⬅️ Muat data terlebih dahulu dari sidebar.")
    else:
        # KPI row
        col1, col2, col3, col4 = st.columns(4)
        sentiment_counts = df_display["sentiment"].value_counts()
        col1.metric("Total Review", f"{len(df_display):,}")
        col2.metric("Positif", sentiment_counts.get("Positif", 0))
        col3.metric("Netral", sentiment_counts.get("Netral", 0))
        col4.metric("Negatif", sentiment_counts.get("Negatif", 0))

        st.markdown("---")

        # Charts row
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            st.markdown("##### Distribusi Rating")
            rating_counts = df_display["rating"].value_counts().sort_index()
            fig_rating = px.bar(
                x=rating_counts.index.astype(str),
                y=rating_counts.values,
                labels={"x": "Rating", "y": "Jumlah"},
                color=rating_counts.index.astype(str),
                color_discrete_sequence=["#ef4444", "#f97316", "#f59e0b", "#84cc16", "#10b981"],
            )
            fig_rating.update_layout(
                showlegend=False,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#ccc",
                margin=dict(l=20, r=20, t=30, b=20),
                xaxis=dict(title="Rating ⭐"),
                yaxis=dict(title="Jumlah Review"),
            )
            st.plotly_chart(fig_rating, use_container_width=True)

        with chart_col2:
            st.markdown("##### Distribusi Sentimen")
            sent_data = df_display["sentiment"].value_counts()
            fig_pie = px.pie(
                names=sent_data.index,
                values=sent_data.values,
                color=sent_data.index,
                color_discrete_map=SENTIMENT_COLORS,
                hole=0.45,
            )
            fig_pie.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#ccc",
                margin=dict(l=20, r=20, t=30, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
            )
            fig_pie.update_traces(textinfo="percent+label", textfont_size=13)
            st.plotly_chart(fig_pie, use_container_width=True)

        # Version chart
        if "version" in df_display.columns:
            st.markdown("##### Jumlah Review per Versi Aplikasi")
            ver_counts = df_display.groupby(["version", "sentiment"]).size().reset_index(name="count")
            fig_ver = px.bar(
                ver_counts, x="version", y="count", color="sentiment",
                color_discrete_map=SENTIMENT_COLORS,
                barmode="stack",
                labels={"version": "Versi", "count": "Jumlah", "sentiment": "Sentimen"},
            )
            fig_ver.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#ccc",
                margin=dict(l=20, r=20, t=30, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
            )
            st.plotly_chart(fig_ver, use_container_width=True)

        # Data table
        st.markdown("##### 📋 Tabel Data Review")
        st.dataframe(
            df_display[["review_text", "rating", "sentiment", "version", "date"]],
            use_container_width=True,
            height=400,
        )


# ──────────────────────────────────────────────────────────────────────────────
#  TAB 2 — PREPROCESSING
# ──────────────────────────────────────────────────────────────────────────────
with tab2:
    df_proc = st.session_state.df_processed
    if df_proc is None:
        st.info("⬅️ Latih model terlebih dahulu untuk melihat hasil preprocessing.")
    else:
        st.markdown("##### Before / After Preprocessing")
        compare_df = df_proc[["review_text", "cleaned_text", "sentiment"]].head(20).copy()
        compare_df.columns = ["Review Asli", "Setelah Preprocessing", "Sentimen"]
        st.dataframe(compare_df, use_container_width=True, height=400)

        st.markdown("---")

        # Word clouds
        st.markdown("##### ☁️ Word Cloud per Sentimen")
        wc_cols = st.columns(3)
        for idx, sent in enumerate(["Positif", "Netral", "Negatif"]):
            texts = " ".join(df_proc[df_proc["sentiment"] == sent]["cleaned_text"].tolist())
            with wc_cols[idx]:
                st.markdown(f"**{sent}**")
                if texts.strip():
                    wc = WordCloud(
                        width=600, height=350,
                        background_color="#1a1a2e",
                        colormap="cool" if sent == "Positif" else ("autumn" if sent == "Negatif" else "Set2"),
                        max_words=80,
                        contour_width=1,
                        contour_color=SENTIMENT_COLORS[sent],
                    ).generate(texts)
                    fig_wc, ax_wc = plt.subplots(figsize=(6, 3.5))
                    ax_wc.imshow(wc, interpolation="bilinear")
                    ax_wc.axis("off")
                    fig_wc.patch.set_facecolor("#0e1117")
                    st.pyplot(fig_wc)
                    plt.close(fig_wc)
                else:
                    st.caption("Tidak ada data.")


# ──────────────────────────────────────────────────────────────────────────────
#  TAB 3 — MODEL & EVALUASI
# ──────────────────────────────────────────────────────────────────────────────
with tab3:
    result = st.session_state.model_result
    if result is None:
        st.info("⬅️ Latih model terlebih dahulu dari sidebar.")
    else:
        # Metrics row
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", f"{result['accuracy']*100:.2f}%")
        m2.metric("Precision", f"{result['precision']*100:.2f}%")
        m3.metric("Recall", f"{result['recall']*100:.2f}%")
        m4.metric("F1-Score", f"{result['f1_score']*100:.2f}%")

        st.markdown("---")

        ev_col1, ev_col2 = st.columns(2)

        # Confusion Matrix
        with ev_col1:
            st.markdown("##### Confusion Matrix")
            cm = result["confusion_matrix"]
            labels = result["labels"]
            fig_cm = px.imshow(
                cm, x=labels, y=labels,
                text_auto=True,
                color_continuous_scale="Purples",
                labels={"x": "Prediksi", "y": "Aktual", "color": "Jumlah"},
            )
            fig_cm.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#ccc",
                margin=dict(l=20, r=20, t=30, b=20),
            )
            st.plotly_chart(fig_cm, use_container_width=True)

        # Classification Report
        with ev_col2:
            st.markdown("##### Classification Report")
            report = result["classification_report"]
            report_df = pd.DataFrame(report).transpose()
            # Keep only class rows + weighted avg
            keep_rows = [r for r in report_df.index if r in labels or r == "weighted avg"]
            report_df = report_df.loc[keep_rows]
            for c in ["precision", "recall", "f1-score"]:
                if c in report_df.columns:
                    report_df[c] = report_df[c].apply(lambda v: f"{v*100:.2f}%")
            if "support" in report_df.columns:
                report_df["support"] = report_df["support"].astype(int)
            st.dataframe(report_df, use_container_width=True)

            st.markdown("---")
            st.markdown(f"**N-Gram:** `{ngram_choice}`")
            st.markdown(f"**Train size:** {result['train_size']}  |  **Test size:** {result['test_size']}")
            st.markdown(f"**Total fitur (features):** {len(result['feature_names']):,}")

        # N-Gram comparison
        st.markdown("---")
        st.markdown("##### 📊 Perbandingan Performa N-Gram")
        comp = st.session_state.ngram_comparison
        if comp is not None:
            st.dataframe(comp, use_container_width=True)
            fig_comp = px.bar(
                comp.melt(id_vars=["N-Gram"], value_vars=["Accuracy", "Precision", "Recall", "F1-Score"]),
                x="N-Gram", y="value", color="variable",
                barmode="group",
                labels={"value": "Skor (%)", "variable": "Metrik"},
                color_discrete_sequence=["#667eea", "#764ba2", "#f093fb", "#10b981"],
            )
            fig_comp.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#ccc",
                margin=dict(l=20, r=20, t=30, b=40),
                legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
            )
            st.plotly_chart(fig_comp, use_container_width=True)
        else:
            st.caption("Klik **Bandingkan N-Gram** di sidebar untuk membandingkan semua konfigurasi.")


# ──────────────────────────────────────────────────────────────────────────────
#  TAB 4 — PREDIKSI
# ──────────────────────────────────────────────────────────────────────────────
with tab4:
    result = st.session_state.model_result
    if result is None:
        st.info("⬅️ Latih model terlebih dahulu dari sidebar.")
    else:
        st.markdown("##### 🔮 Prediksi Sentimen Teks Manual")
        user_text = st.text_area(
            "Masukkan teks review untuk diprediksi",
            placeholder="Contoh: Game ini sangat seru dan grafisnya bagus banget!",
            height=120,
        )

        if st.button("🔍 Prediksi", use_container_width=True, type="primary"):
            if user_text.strip():
                cleaned = preprocess(user_text, use_stemming=use_stemming)
                pred, proba = predict_text(cleaned, result["model"], result["vectorizer"])

                st.markdown("---")

                # Result display
                badge_class = f"badge-{pred.lower()}"
                st.markdown(f"""
                <div style="text-align:center; padding:24px 0;">
                    <p style="font-size:1rem; color:#aaa; margin-bottom:8px;">Hasil Prediksi</p>
                    <span class="badge {badge_class}" style="font-size:1.5rem; padding:10px 32px;">
                        {pred}
                    </span>
                </div>
                """, unsafe_allow_html=True)

                # Probability bars
                st.markdown("##### Probabilitas per Kelas")
                prob_cols = st.columns(len(proba))
                for i, (cls, pct) in enumerate(sorted(proba.items())):
                    with prob_cols[i]:
                        st.metric(cls, f"{pct:.1f}%")
                        st.progress(pct / 100)

                # Cleaned text
                with st.expander("🔍 Detail Preprocessing"):
                    st.markdown(f"**Teks asli:** {user_text}")
                    st.markdown(f"**Setelah preprocessing:** `{cleaned}`")
            else:
                st.warning("Masukkan teks terlebih dahulu.")

        # Quick examples
        st.markdown("---")
        st.markdown("##### 💡 Contoh Teks")
        examples = [
            "Game terbaik! Grafisnya keren banget dan gameplay seru abis!",
            "Lumayan buat ngisi waktu, tapi agak lag di HP kentang",
            "Sampah! Penuh cheater dan bug! Uninstall!",
        ]
        for ex in examples:
            if st.button(f"📝 {ex[:60]}...", key=f"ex_{hash(ex)}"):
                cleaned = preprocess(ex, use_stemming=use_stemming)
                pred, proba = predict_text(cleaned, result["model"], result["vectorizer"])
                badge_class = f"badge-{pred.lower()}"
                st.markdown(f'<span class="badge {badge_class}">{pred}</span> — {ex}', unsafe_allow_html=True)
                st.json(proba)
