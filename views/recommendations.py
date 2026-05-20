"""Developer Recommendations Page — auto-generated insights from sentiment data."""
import streamlit as st
import pandas as pd
from collections import Counter

SENTIMENT_COLORS = {"Positif": "#10b981", "Netral": "#f59e0b", "Negatif": "#ef4444"}

# Keyword categories for automatic insight generation
COMPLAINT_CATEGORIES = {
    "Performa & Lag": ["lag", "lemot", "lambat", "crash", "force", "close", "hang", "patah"],
    "Cheater & Hacker": ["cheat", "cheater", "hacker", "hack", "bot", "aim", "wall"],
    "Bug & Error": ["bug", "error", "glitch", "rusak", "masalah", "broken"],
    "Server & Koneksi": ["server", "ping", "koneksi", "disconnect", "down", "maintenance", "desync"],
    "Matchmaking": ["matchmaking", "match", "rank", "ranking", "adil", "balance"],
    "UI & Kontrol": ["kontrol", "control", "ui", "sensitivitas", "loading", "interface"],
    "Monetisasi": ["mahal", "bayar", "skin", "uang", "pay", "harga", "boros"],
    "Ukuran & Storage": ["ukuran", "storage", "berat", "besar", "download", "baterai", "panas"],
}

POSITIVE_CATEGORIES = {
    "Gameplay Seru": ["seru", "asik", "asyik", "menyenangkan", "adiktif", "ketagihan"],
    "Grafis Bagus": ["grafis", "grafik", "visual", "detail", "realistis", "hd", "bagus"],
    "Update Positif": ["update", "fitur", "mode", "baru", "konten", "event"],
    "Performa Lancar": ["lancar", "smooth", "stabil", "cepat", "responsif", "fps"],
    "Sosial & Mabar": ["teman", "squad", "mabar", "voice", "chat", "tim", "team"],
}


def _count_keywords(texts, categories):
    """Count keyword occurrences in texts for each category."""
    all_words = " ".join(texts).lower().split()
    word_counts = Counter(all_words)
    results = {}
    for cat, keywords in categories.items():
        total = sum(word_counts.get(kw, 0) for kw in keywords)
        if total > 0:
            results[cat] = total
    return dict(sorted(results.items(), key=lambda x: x[1], reverse=True))


def render(t):
    """Render the Developer Recommendations page."""
    st.markdown(f"### 💡 {t.get('nav_recommendations', 'Rekomendasi Developer')}")
    st.markdown(f"_{t.get('rec_desc', 'Insight otomatis dan rekomendasi berdasarkan analisis sentimen review.')}_")

    df = st.session_state.get("df_raw")
    df_proc = st.session_state.get("df_processed")

    if df is None or len(df) == 0:
        from views import redirect_to_upload
        redirect_to_upload(t)
        return

    # ── Overall Sentiment Summary ──
    st.markdown(f"##### 📊 {t.get('rec_summary', 'Ringkasan Sentimen')}")
    sentiment_counts = df["sentiment"].value_counts()
    total = len(df)

    s1, s2, s3 = st.columns(3)
    for col, sent, color in [(s1, "Positif", "#10b981"), (s2, "Netral", "#f59e0b"), (s3, "Negatif", "#ef4444")]:
        count = sentiment_counts.get(sent, 0)
        pct = count / total * 100 if total > 0 else 0
        with col:
            st.markdown(f"""
            <div class="glass-card" style="padding:20px; text-align:center; border-top:3px solid {color};">
                <h3 style="margin:0; color:{color};">{pct:.1f}%</h3>
                <p style="margin:4px 0 0; opacity:.7;">{sent} ({count:,})</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Determine text source ──
    if df_proc is not None:
        neg_texts = df_proc[df_proc["sentiment"] == "Negatif"]["cleaned_text"].tolist()
        pos_texts = df_proc[df_proc["sentiment"] == "Positif"]["cleaned_text"].tolist()
        neu_texts = df_proc[df_proc["sentiment"] == "Netral"]["cleaned_text"].tolist()
    else:
        neg_texts = df[df["sentiment"] == "Negatif"]["review_text"].astype(str).tolist()
        pos_texts = df[df["sentiment"] == "Positif"]["review_text"].astype(str).tolist()
        neu_texts = df[df["sentiment"] == "Netral"]["review_text"].astype(str).tolist()

    # ── Top Complaints ──
    complaint_counts = _count_keywords(neg_texts, COMPLAINT_CATEGORIES)

    if complaint_counts:
        st.markdown(f"##### 🚨 {t.get('rec_complaints', 'Keluhan Utama (dari Review Negatif)')}")

        for i, (cat, count) in enumerate(complaint_counts.items()):
            pct = count / max(len(" ".join(neg_texts).split()), 1) * 100
            severity = "🔴" if pct > 2 else "🟡" if pct > 0.5 else "🟢"
            st.markdown(f"""
            <div class="glass-card" style="padding:14px 20px; margin-bottom:8px; display:flex; align-items:center; gap:12px;">
                <span style="font-size:1.3rem;">{severity}</span>
                <div style="flex:1;">
                    <strong>{cat}</strong>
                    <span style="opacity:.6; font-size:.85rem; margin-left:8px;">({count} kemunculan kata kunci)</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

    # ── Positive Highlights ──
    positive_counts = _count_keywords(pos_texts, POSITIVE_CATEGORIES)

    if positive_counts:
        st.markdown(f"##### 🌟 {t.get('rec_highlights', 'Hal yang Disukai Pengguna')}")

        highlight_cols = st.columns(min(len(positive_counts), 3))
        for i, (cat, count) in enumerate(positive_counts.items()):
            with highlight_cols[i % 3]:
                st.markdown(f"""
                <div class="glass-card" style="padding:18px; text-align:center; border-top:3px solid #10b981;">
                    <h4 style="margin:0 0 6px; color:#10b981;">✅ {cat}</h4>
                    <p style="margin:0; opacity:.7; font-size:.85rem;">{count} kemunculan</p>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")

    # ── Actionable Recommendations ──
    st.markdown(f"##### 💡 {t.get('rec_actions', 'Rekomendasi Aksi untuk Developer')}")

    recommendations = []
    neg_pct = sentiment_counts.get("Negatif", 0) / total * 100 if total > 0 else 0

    if neg_pct > 30:
        recommendations.append(("🔴", t.get("rec_high_neg", "Tingkat sentimen negatif tinggi (>30%). Prioritaskan perbaikan bug dan performa.")))
    elif neg_pct > 15:
        recommendations.append(("🟡", t.get("rec_mid_neg", "Tingkat sentimen negatif moderat (15-30%). Fokus pada feedback negatif utama.")))
    else:
        recommendations.append(("🟢", t.get("rec_low_neg", "Sentimen secara keseluruhan positif. Pertahankan kualitas dan terus berinovasi.")))

    if "Performa & Lag" in complaint_counts:
        recommendations.append(("⚡", t.get("rec_perf", "Optimasi performa game untuk perangkat mid-range dan low-end.")))
    if "Cheater & Hacker" in complaint_counts:
        recommendations.append(("🛡️", t.get("rec_cheat", "Tingkatkan sistem anti-cheat dan percepat proses report cheater.")))
    if "Bug & Error" in complaint_counts:
        recommendations.append(("🐛", t.get("rec_bug", "Lakukan QA testing menyeluruh sebelum release update.")))
    if "Server & Koneksi" in complaint_counts:
        recommendations.append(("🌐", t.get("rec_server", "Tingkatkan infrastruktur server, terutama di region Asia Tenggara.")))
    if "Monetisasi" in complaint_counts:
        recommendations.append(("💰", t.get("rec_monetize", "Review ulang model monetisasi. Pertimbangkan menawarkan lebih banyak konten gratis.")))

    for icon, text in recommendations:
        st.markdown(f"""
        <div class="glass-card" style="padding:16px 20px; margin-bottom:8px; display:flex; align-items:flex-start; gap:12px;">
            <span style="font-size:1.4rem;">{icon}</span>
            <p style="margin:0; line-height:1.5;">{text}</p>
        </div>
        """, unsafe_allow_html=True)

    # ── Sample Negative Reviews ──
    st.markdown("---")
    st.markdown(f"##### 📝 {t.get('rec_sample_neg', 'Contoh Review Negatif Teratas')}")
    neg_df = df[df["sentiment"] == "Negatif"][["review_text", "rating"]].head(5)
    if len(neg_df) > 0:
        st.dataframe(neg_df, use_container_width=True)
    else:
        st.caption(t.get("no_data", "Tidak ada data."))
