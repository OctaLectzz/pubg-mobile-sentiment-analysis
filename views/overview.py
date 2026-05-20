"""Overview / Dashboard Home Page."""
import streamlit as st
import plotly.express as px

SENTIMENT_COLORS = {"Positif": "#10b981", "Netral": "#f59e0b", "Negatif": "#ef4444"}


def render(t):
    """Render the Overview dashboard page."""
    # Hero Banner
    st.markdown(f"""
    <div class="hero-banner">
        <h1>{t['hero_title']}</h1>
        <p>{t['hero_desc']}</p>
    </div>
    """, unsafe_allow_html=True)

    df = st.session_state.get("df_raw")
    model = st.session_state.get("model_result")

    if df is None or len(df) == 0:
        # ── Quick Start Guide ──
        st.markdown(f"### 🚀 {t.get('quickstart_title', 'Mulai Cepat')}")
        st.markdown(f"_{t.get('quickstart_desc', 'Ikuti langkah berikut untuk memulai analisis sentimen:')}_")

        cols = st.columns(3)
        steps = [
            ("📂", t.get("step1_title", "Muat Data"),
             t.get("step1_desc", "Buka menu Upload Dataset untuk memuat data sample, scrape dari App Store, atau upload CSV.")),
            ("🧠", t.get("step2_title", "Latih Model"),
             t.get("step2_desc", "Konfigurasi N-Gram di sidebar, lalu latih model Naïve Bayes dari menu Klasifikasi.")),
            ("📊", t.get("step3_title", "Analisis Hasil"),
             t.get("step3_desc", "Jelajahi evaluasi model, visualisasi, word cloud, dan prediksi sentimen.")),
        ]
        for i, (icon, title, desc) in enumerate(steps):
            with cols[i]:
                st.markdown(f"""
                <div class="glass-card" style="text-align:center; padding:28px 20px;">
                    <div style="font-size:2.8rem; margin-bottom:14px;">{icon}</div>
                    <h4 style="margin:0 0 10px; font-weight:700;">{title}</h4>
                    <p style="opacity:.7; font-size:.88rem; margin:0; line-height:1.5;">{desc}</p>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.info(t.get("quickstart_info", "👈 Gunakan menu **📂 Upload Dataset** di sidebar untuk memuat data."))
        return

    # ── Data Loaded — Dashboard Metrics ──
    sentiment_counts = df["sentiment"].value_counts()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(t["total_reviews"], f"{len(df):,}")
    col2.metric(t["positive"], sentiment_counts.get("Positif", 0))
    col3.metric(t["neutral"], sentiment_counts.get("Netral", 0))
    col4.metric(t["negative"], sentiment_counts.get("Negatif", 0))

    st.markdown("---")

    # ── Charts Row ──
    chart1, chart2 = st.columns(2)

    with chart1:
        st.markdown(f"##### {t.get('sentiment_dist', 'Distribusi Sentimen')}")
        fig = px.pie(
            names=sentiment_counts.index,
            values=sentiment_counts.values,
            color=sentiment_counts.index,
            color_discrete_map=SENTIMENT_COLORS,
            hole=0.45,
        )
        fig.update_layout(
            margin=dict(l=20, r=20, t=30, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        )
        fig.update_traces(textinfo="percent+label", textfont_size=13)
        st.plotly_chart(fig, use_container_width=True)

    with chart2:
        st.markdown(f"##### {t.get('rating_dist', 'Distribusi Rating')}")
        rating_counts = df["rating"].value_counts().sort_index()
        fig = px.bar(
            x=rating_counts.index.astype(str), y=rating_counts.values,
            labels={"x": "Rating ⭐", "y": t.get("count_label", "Jumlah")},
            color=rating_counts.index.astype(str),
            color_discrete_sequence=["#ef4444", "#f97316", "#f59e0b", "#84cc16", "#10b981"],
        )
        fig.update_layout(
            showlegend=False, margin=dict(l=20, r=20, t=30, b=20),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Model Status ──
    st.markdown("---")
    if model:
        mcol1, mcol2, mcol3 = st.columns(3)
        mcol1.metric("Accuracy", f"{model['accuracy']*100:.1f}%")
        mcol2.metric("F1-Score", f"{model['f1_score']*100:.1f}%")
        mcol3.metric(t.get("total_features", "Total Fitur"), f"{len(model['feature_names']):,}")
    else:
        st.warning(t.get("model_not_trained", "⚠️ Model belum dilatih. Buka menu **🧠 Klasifikasi Sentimen** untuk melatih model."))

    # ── Recent Reviews ──
    st.markdown("---")
    st.markdown(f"##### {t.get('recent_reviews', 'Review Terbaru')}")
    display_cols = [c for c in ["review_text", "rating", "sentiment", "date"] if c in df.columns]
    st.dataframe(df[display_cols].head(8), use_container_width=True)
