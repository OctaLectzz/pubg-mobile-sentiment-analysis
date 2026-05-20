"""Data Exploration Page — charts, metrics, and data table."""
import streamlit as st
import plotly.express as px
from utils.export import export_csv, export_data_pdf

SENTIMENT_COLORS = {"Positif": "#10b981", "Netral": "#f59e0b", "Negatif": "#ef4444"}


def render(t):
    """Render the Data Exploration page."""
    st.markdown(f"### 🔍 {t.get('nav_exploration', 'Eksplorasi Data')}")

    df = st.session_state.get("df_raw")
    if df is None or len(df) == 0:
        from views import redirect_to_upload
        redirect_to_upload(t)
        return

    # ── Metric Cards ──
    sentiment_counts = df["sentiment"].value_counts()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(t["total_reviews"], f"{len(df):,}")
    col2.metric(t["positive"], sentiment_counts.get("Positif", 0))
    col3.metric(t["neutral"], sentiment_counts.get("Netral", 0))
    col4.metric(t["negative"], sentiment_counts.get("Negatif", 0))

    st.markdown("---")

    # ── Export Buttons ──
    exp_col1, exp_col2 = st.columns(2)
    with exp_col1:
        csv_bytes = export_csv(df)
        st.download_button(t["export_csv"], csv_bytes, "pubg_reviews.csv", "text/csv", use_container_width=True)
    with exp_col2:
        pdf_bytes = export_data_pdf(df)
        st.download_button(t["export_pdf"], pdf_bytes, "pubg_report.pdf", "application/pdf", use_container_width=True)

    st.markdown("---")

    # ── Charts Row ──
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown(f"##### {t['rating_dist']}")
        rating_counts = df["rating"].value_counts().sort_index()
        fig_rating = px.bar(
            x=rating_counts.index.astype(str), y=rating_counts.values,
            labels={"x": "Rating ⭐", "y": t["count_label"]},
            color=rating_counts.index.astype(str),
            color_discrete_sequence=["#ef4444", "#f97316", "#f59e0b", "#84cc16", "#10b981"],
        )
        fig_rating.update_layout(
            showlegend=False, margin=dict(l=20, r=20, t=30, b=20),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_rating, use_container_width=True)

    with chart_col2:
        st.markdown(f"##### {t['sentiment_dist']}")
        fig_pie = px.pie(
            names=sentiment_counts.index, values=sentiment_counts.values,
            color=sentiment_counts.index, color_discrete_map=SENTIMENT_COLORS, hole=0.45,
        )
        fig_pie.update_layout(
            margin=dict(l=20, r=20, t=30, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        )
        fig_pie.update_traces(textinfo="percent+label", textfont_size=13)
        st.plotly_chart(fig_pie, use_container_width=True)

    # ── Reviews per Version ──
    if "version" in df.columns:
        st.markdown(f"##### {t['reviews_per_version']}")
        ver_counts = df.groupby(["version", "sentiment"]).size().reset_index(name="count")
        fig_ver = px.bar(
            ver_counts, x="version", y="count", color="sentiment",
            color_discrete_map=SENTIMENT_COLORS, barmode="stack",
        )
        fig_ver.update_layout(
            margin=dict(l=20, r=20, t=30, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_ver, use_container_width=True)

    # ── Data Table ──
    st.markdown(f"##### {t['data_table']}")
    display_cols = [c for c in ["review_text", "rating", "sentiment", "version", "date", "country"] if c in df.columns]
    st.dataframe(df[display_cols], use_container_width=True, height=400)
