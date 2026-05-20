"""Visualization Page — consolidated charts and interactive analysis."""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter

SENTIMENT_COLORS = {"Positif": "#10b981", "Netral": "#f59e0b", "Negatif": "#ef4444"}


def render(t):
    """Render the Visualization page."""
    st.markdown(f"### 📊 {t.get('nav_visualization', 'Visualisasi')}")
    st.markdown(f"_{t.get('viz_desc', 'Visualisasi interaktif untuk eksplorasi data sentimen secara mendalam.')}_")

    df = st.session_state.get("df_raw")
    if df is None or len(df) == 0:
        from views import redirect_to_upload
        redirect_to_upload(t)
        return

    # ── Sentiment Trend Over Time ──
    if "date" in df.columns:
        st.markdown(f"##### 📅 {t.get('viz_trend', 'Tren Sentimen Berdasarkan Waktu')}")
        df_date = df.copy()
        df_date["date"] = pd.to_datetime(df_date["date"], errors="coerce")
        df_date = df_date.dropna(subset=["date"])

        if len(df_date) > 0:
            trend = df_date.groupby([df_date["date"].dt.to_period("M").astype(str), "sentiment"]).size()
            trend = trend.reset_index(name="count")
            trend.columns = ["Bulan", "Sentimen", "Jumlah"]

            fig_trend = px.line(
                trend, x="Bulan", y="Jumlah", color="Sentimen",
                color_discrete_map=SENTIMENT_COLORS, markers=True,
            )
            fig_trend.update_layout(
                margin=dict(l=20, r=20, t=30, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                xaxis_title="Periode", yaxis_title=t.get("count_label", "Jumlah"),
            )
            st.plotly_chart(fig_trend, use_container_width=True)

        st.markdown("---")

    # ── Rating vs Sentiment Heatmap ──
    st.markdown(f"##### 🔥 {t.get('viz_heatmap', 'Korelasi Rating dan Sentimen')}")

    cross = pd.crosstab(df["rating"], df["sentiment"])
    for s in ["Positif", "Netral", "Negatif"]:
        if s not in cross.columns:
            cross[s] = 0
    cross = cross[["Positif", "Netral", "Negatif"]]

    fig_heat = px.imshow(
        cross.values, x=cross.columns.tolist(), y=cross.index.astype(str).tolist(),
        text_auto=True, color_continuous_scale="YlOrRd",
        labels={"x": "Sentimen", "y": "Rating", "color": t.get("count_label", "Jumlah")},
    )
    fig_heat.update_layout(
        margin=dict(l=20, r=20, t=30, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown("---")

    # ── Country Breakdown (if multi-country) ──
    if "country" in df.columns and df["country"].nunique() > 1:
        st.markdown(f"##### 🌍 {t.get('viz_country', 'Distribusi Sentimen per Negara')}")

        country_sent = df.groupby(["country", "sentiment"]).size().reset_index(name="count")
        fig_country = px.bar(
            country_sent, x="country", y="count", color="sentiment",
            color_discrete_map=SENTIMENT_COLORS, barmode="group",
            labels={"country": t.get("country_label", "Negara"), "count": t.get("count_label", "Jumlah")},
        )
        fig_country.update_layout(
            margin=dict(l=20, r=20, t=30, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_country, use_container_width=True)
        st.markdown("---")

    # ── Top Words per Sentiment ──
    df_proc = st.session_state.get("df_processed")
    if df_proc is not None:
        st.markdown(f"##### 🔤 {t.get('viz_topwords', 'Kata Teratas per Sentimen')}")

        top_n = st.slider(t.get("topwords_slider", "Jumlah kata teratas"), 5, 25, 10, key="viz_topn")

        word_cols = st.columns(3)
        for idx, sent in enumerate(["Positif", "Netral", "Negatif"]):
            with word_cols[idx]:
                texts = " ".join(df_proc[df_proc["sentiment"] == sent]["cleaned_text"].tolist())
                words = texts.split()
                counter = Counter(words)
                top = counter.most_common(top_n)
                if top:
                    wdf = pd.DataFrame(top, columns=["Kata", "Frekuensi"])
                    fig_w = px.bar(
                        wdf, x="Frekuensi", y="Kata", orientation="h",
                        color_discrete_sequence=[SENTIMENT_COLORS[sent]],
                    )
                    fig_w.update_layout(
                        yaxis=dict(autorange="reversed"),
                        margin=dict(l=10, r=10, t=30, b=10),
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        height=max(250, top_n * 28), title=sent,
                    )
                    st.plotly_chart(fig_w, use_container_width=True)
                else:
                    st.caption(t.get("no_data", "Tidak ada data."))
    else:
        from views import redirect_to_train
        redirect_to_train(t)

    # ── Rating Distribution by Sentiment (Violin) ──
    st.markdown("---")
    st.markdown(f"##### 🎻 {t.get('viz_violin', 'Distribusi Rating per Sentimen')}")
    fig_violin = px.violin(
        df, x="sentiment", y="rating", color="sentiment",
        color_discrete_map=SENTIMENT_COLORS, box=True, points="all",
        labels={"sentiment": "Sentimen", "rating": "Rating"},
    )
    fig_violin.update_layout(
        showlegend=False, margin=dict(l=20, r=20, t=30, b=20),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_violin, use_container_width=True)
