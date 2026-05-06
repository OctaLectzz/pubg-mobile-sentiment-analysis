"""
Streamlit Dashboard — PUBG Mobile Sentiment Analysis
Naïve Bayes Classifier with N-Gram Feature Extraction
"""
import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from wordcloud import WordCloud

from scraper import generate_sample_data, scrape_reviews, scrape_multiple_countries, OUTPUT_FILE, COUNTRY_OPTIONS
from preprocessing import preprocess, preprocess_dataframe
from model import build_model, predict_text, predict_raw_text, compare_ngrams, save_model, NGRAM_OPTIONS
from config.languages import get_lang, LANGUAGES
from utils.export import export_csv, export_data_pdf, export_model_pdf

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PUBG Mobile Sentiment Analysis",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── CSS (supports light & dark mode) ────────────────────────────────────────
def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* ── metric cards (adaptive) ── */
    div[data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,.15);
        border-radius: 14px;
        padding: 18px 22px;
        box-shadow: 0 2px 12px rgba(0,0,0,.08);
        transition: transform .2s, box-shadow .2s;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,.12);
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-size: 1.8rem !important; font-weight: 700;
    }

    /* ── tabs ── */
    button[data-baseweb="tab"] {
        font-weight: 600 !important; font-size: .95rem !important;
        border-radius: 10px 10px 0 0 !important;
    }

    /* ── tables ── */
    .stDataFrame { border-radius: 12px; overflow: hidden; }

    /* ── hero banner ── */
    .hero-banner {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        border-radius: 16px;
        padding: 32px 36px;
        margin-bottom: 24px;
        color: white;
        box-shadow: 0 8px 32px rgba(102,126,234,.35);
        animation: heroGlow 4s ease-in-out infinite alternate;
    }
    @keyframes heroGlow {
        0% { box-shadow: 0 8px 32px rgba(102,126,234,.35); }
        100% { box-shadow: 0 8px 40px rgba(118,75,162,.45); }
    }
    .hero-banner h1 { margin: 0 0 6px 0; font-size: 2rem; font-weight: 800; }
    .hero-banner p  { margin: 0; opacity: .88; font-size: 1.05rem; }

    /* ── sentiment badges ── */
    .badge { display:inline-block; padding:4px 14px; border-radius:20px; font-weight:600; font-size:.82rem; }
    .badge-positif  { background:#10b981; color:#fff; }
    .badge-netral   { background:#f59e0b; color:#fff; }
    .badge-negatif  { background:#ef4444; color:#fff; }

    /* ── sidebar branding ── */
    section[data-testid="stSidebar"] > div:first-child { padding-top: 1rem; }

    /* ── full-screen loading overlay ── */
    .loading-overlay {
        position: fixed; top: 0; left: 0; width: 100vw; height: 100vh;
        background: rgba(0,0,0,.55); backdrop-filter: blur(4px);
        z-index: 9999999; display: flex; align-items: center; justify-content: center;
        flex-direction: column; gap: 16px;
    }
    .loading-overlay .spinner {
        width: 48px; height: 48px;
        border: 4px solid rgba(255,255,255,.3);
        border-top: 4px solid #667eea;
        border-radius: 50%;
        animation: spin .8s linear infinite;
    }
    @keyframes spin { to { transform: rotate(360deg); } }
    .loading-overlay p { color: #fff; font-size: 1.1rem; font-weight: 600; margin: 0; }
    </style>
    """, unsafe_allow_html=True)

inject_css()

# ── Colour palette ───────────────────────────────────────────────────────────
SENTIMENT_COLORS = {"Positif": "#10b981", "Netral": "#f59e0b", "Negatif": "#ef4444"}


# ── Session state ────────────────────────────────────────────────────────────
def init_session_state():
    defaults = {
        "df_raw": None,
        "df_processed": None,
        "model_result": None,
        "ngram_comparison": None,
        "lang": "en",
        "use_stemming": True,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session_state()

# ── Loading overlay helper ───────────────────────────────────────────────────
def show_loading(message="Processing..."):
    """Show a full-screen loading overlay."""
    st.markdown(f"""
    <div class="loading-overlay" id="loadingOverlay">
        <div class="spinner"></div>
        <p>{message}</p>
    </div>
    """, unsafe_allow_html=True)

# ── Language helper ──────────────────────────────────────────────────────────
t = get_lang(st.session_state.lang)


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    # Language selector at top
    lang_options = {v["lang_name"]: k for k, v in LANGUAGES.items()}
    selected_lang_name = st.selectbox(
        t["language_label"],
        list(lang_options.keys()),
        index=list(lang_options.values()).index(st.session_state.lang),
    )
    new_lang = lang_options[selected_lang_name]
    if new_lang != st.session_state.lang:
        st.session_state.lang = new_lang
        st.rerun()

    st.markdown(f"## 🎮 {t['sidebar_title']}")
    st.markdown(f"### {t['sidebar_subtitle']}")
    st.markdown("---")

    # ── Data source ──
    st.markdown(f"#### 📂 {t['data_source']}")
    data_source = st.radio(
        t["data_source"],
        [t["data_source_sample"], t["data_source_scrape"], t["data_source_upload"]],
        label_visibility="collapsed",
    )

    if data_source == t["data_source_upload"]:
        uploaded = st.file_uploader(t["upload_csv"], type=["csv"])
        if uploaded is not None:
            st.session_state.df_raw = pd.read_csv(uploaded)
            st.success(f"✅ {len(st.session_state.df_raw)} {t['rows_loaded']}")

    elif data_source == t["data_source_scrape"]:
        select_all = st.checkbox(t["select_all"], value=False)
        country_names = list(COUNTRY_OPTIONS.keys())
        if select_all:
            selected_countries = st.multiselect(
                t["select_countries"], country_names, default=country_names
            )
        else:
            selected_countries = st.multiselect(
                t["select_countries"], country_names, default=["🇮🇩 Indonesia"]
            )
        scrape_count = st.number_input(t["review_count"], min_value=50, max_value=2000, value=200, step=50)

        if st.button(t["scrape_btn"], use_container_width=True, type="primary"):
            if selected_countries:
                country_codes = [COUNTRY_OPTIONS[c] for c in selected_countries]
                country_str = ", ".join(selected_countries[:3]) + ("..." if len(selected_countries) > 3 else "")
                msg = t["scrape_spinner"].format(count=scrape_count, countries=country_str)
                show_loading(msg)
                with st.spinner(msg):
                    if len(country_codes) == 1:
                        df_scraped = scrape_reviews(country=country_codes[0], how_many=scrape_count)
                    else:
                        per_country = max(50, scrape_count // len(country_codes) + 10) # fetch a bit extra per country just in case
                        df_scraped = scrape_multiple_countries(country_codes, how_many_per_country=per_country)
                        
                    # Truncate to exact requested count if we got more
                    if df_scraped is not None and not df_scraped.empty and len(df_scraped) > scrape_count:
                        # Shuffle to ensure mix of countries if we truncate
                        df_scraped = df_scraped.sample(n=scrape_count, random_state=42).reset_index(drop=True)
                if df_scraped is not None and not df_scraped.empty:
                    st.session_state.df_raw = df_scraped
                    st.success(f"✅ {len(df_scraped)} {t['scrape_success']}")
                    st.rerun()
                else:
                    st.error(t["scrape_fail"])

    else:
        if st.button(t["load_sample"], use_container_width=True):
            with st.spinner(t["load_sample_spinner"]):
                generate_sample_data()
            st.session_state.df_raw = pd.read_csv(OUTPUT_FILE)
            st.success(f"✅ {len(st.session_state.df_raw)} {t['load_sample_success']}")

        if st.session_state.df_raw is None and os.path.exists(OUTPUT_FILE):
            st.session_state.df_raw = pd.read_csv(OUTPUT_FILE)

    st.markdown("---")

    # ── Version filter ──
    if st.session_state.df_raw is not None and "version" in st.session_state.df_raw.columns:
        versions = sorted(st.session_state.df_raw["version"].dropna().unique().tolist())
        selected_versions = st.multiselect(t["filter_version"], options=versions, default=versions)
    else:
        selected_versions = None

    st.markdown("---")

    # ── N-Gram selector ──
    st.markdown(f"#### {t['model_config']}")
    ngram_choice = st.selectbox(t["ngram_type"], list(NGRAM_OPTIONS.keys()), index=3)
    use_stemming = st.checkbox(t["use_stemming"], value=True)
    st.session_state.use_stemming = use_stemming
    test_size = st.slider(t["test_size"], 10, 40, 20, 5) / 100

    st.markdown("---")

    # ── Train buttons ──
    btn_train = st.button(t["train_btn"], use_container_width=True, type="primary")
    btn_compare = st.button(t["compare_btn"], use_container_width=True)

    st.markdown("---")
    st.caption(t["copyright"])


# ══════════════════════════════════════════════════════════════════════════════
#  APPLY VERSION FILTER
# ══════════════════════════════════════════════════════════════════════════════
df_display = st.session_state.df_raw
if df_display is not None and selected_versions is not None:
    df_display = df_display[df_display["version"].isin(selected_versions)]


# ══════════════════════════════════════════════════════════════════════════════
#  TRAINING LOGIC
# ══════════════════════════════════════════════════════════════════════════════
if btn_train and df_display is not None and len(df_display) > 0:
    show_loading(t["train_spinner"])
    with st.spinner(t["preprocess_spinner"]):
        df_proc = preprocess_dataframe(df_display, use_stemming=use_stemming)
        st.session_state.df_processed = df_proc
    with st.spinner(t["train_spinner"]):
        ngram_range = NGRAM_OPTIONS[ngram_choice]
        result = build_model(df_proc["cleaned_text"], df_proc["sentiment"], ngram_range=ngram_range, test_size=test_size)
        st.session_state.model_result = result
        save_model(result["model"], result["vectorizer"])
    st.toast(t["train_success"], icon="🎉")
    st.rerun()

if btn_compare and df_display is not None and len(df_display) > 0:
    show_loading(t["compare_spinner"])
    with st.spinner(t["preprocess_spinner"]):
        df_proc = preprocess_dataframe(df_display, use_stemming=use_stemming)
        st.session_state.df_processed = df_proc
    with st.spinner(t["compare_spinner"]):
        comparison = compare_ngrams(df_proc["cleaned_text"], df_proc["sentiment"], test_size=test_size)
        st.session_state.ngram_comparison = comparison
    st.toast(t["compare_success"], icon="📊")
    st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
#  HERO BANNER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="hero-banner">
    <h1>{t['hero_title']}</h1>
    <p>{t['hero_desc']}</p>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([t["tab_explore"], t["tab_preprocess"], t["tab_model"], t["tab_predict"]])


# ── TAB 1 — DATA EXPLORATION ────────────────────────────────────────────────
with tab1:
    if df_display is None or len(df_display) == 0:
        st.info(t["load_data_first"])
    else:
        col1, col2, col3, col4 = st.columns(4)
        sentiment_counts = df_display["sentiment"].value_counts()
        col1.metric(t["total_reviews"], f"{len(df_display):,}")
        col2.metric(t["positive"], sentiment_counts.get("Positif", 0))
        col3.metric(t["neutral"], sentiment_counts.get("Netral", 0))
        col4.metric(t["negative"], sentiment_counts.get("Negatif", 0))

        st.markdown("---")

        # Export buttons (full width)
        exp_col1, exp_col2 = st.columns(2)
        with exp_col1:
            csv_bytes = export_csv(df_display)
            st.download_button(t["export_csv"], csv_bytes, "pubg_reviews.csv", "text/csv", use_container_width=True)
        with exp_col2:
            pdf_bytes = export_data_pdf(df_display)
            st.download_button(t["export_pdf"], pdf_bytes, "pubg_report.pdf", "application/pdf", use_container_width=True)

        st.markdown("---")

        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            st.markdown(f"##### {t['rating_dist']}")
            rating_counts = df_display["rating"].value_counts().sort_index()
            fig_rating = px.bar(
                x=rating_counts.index.astype(str), y=rating_counts.values,
                labels={"x": "Rating", "y": t["count_label"]},
                color=rating_counts.index.astype(str),
                color_discrete_sequence=["#ef4444", "#f97316", "#f59e0b", "#84cc16", "#10b981"],
            )
            fig_rating.update_layout(showlegend=False, margin=dict(l=20, r=20, t=30, b=20),
                                     xaxis=dict(title="Rating ⭐"), yaxis=dict(title=t["count_label"]))
            st.plotly_chart(fig_rating, use_container_width=True)

        with chart_col2:
            st.markdown(f"##### {t['sentiment_dist']}")
            sent_data = df_display["sentiment"].value_counts()
            fig_pie = px.pie(names=sent_data.index, values=sent_data.values,
                             color=sent_data.index, color_discrete_map=SENTIMENT_COLORS, hole=0.45)
            fig_pie.update_layout(margin=dict(l=20, r=20, t=30, b=20),
                                  legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5))
            fig_pie.update_traces(textinfo="percent+label", textfont_size=13)
            st.plotly_chart(fig_pie, use_container_width=True)

        if "version" in df_display.columns:
            st.markdown(f"##### {t['reviews_per_version']}")
            ver_counts = df_display.groupby(["version", "sentiment"]).size().reset_index(name="count")
            fig_ver = px.bar(ver_counts, x="version", y="count", color="sentiment",
                             color_discrete_map=SENTIMENT_COLORS, barmode="stack")
            fig_ver.update_layout(margin=dict(l=20, r=20, t=30, b=20),
                                  legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5))
            st.plotly_chart(fig_ver, use_container_width=True)

        st.markdown(f"##### {t['data_table']}")
        display_cols = [c for c in ["review_text", "rating", "sentiment", "version", "date", "country"] if c in df_display.columns]
        st.dataframe(df_display[display_cols], use_container_width=True, height=400)


# ── TAB 2 — PREPROCESSING ───────────────────────────────────────────────────
with tab2:
    df_proc = st.session_state.df_processed
    if df_proc is None:
        st.info(t["train_first"])
    else:
        st.markdown(f"##### {t['before_after']}")
        compare_df = df_proc[["review_text", "cleaned_text", "sentiment"]].head(20).copy()
        compare_df.columns = [t["original_review"], t["after_preprocess"], t["sentiment"]]
        st.dataframe(compare_df, use_container_width=True, height=400)

        st.markdown("---")
        st.markdown(f"##### {t['wordcloud_title']}")
        wc_cols = st.columns(3)
        for idx, sent in enumerate(["Positif", "Netral", "Negatif"]):
            texts = " ".join(df_proc[df_proc["sentiment"] == sent]["cleaned_text"].tolist())
            with wc_cols[idx]:
                st.markdown(f"**{sent}**")
                if texts.strip():
                    wc = WordCloud(width=600, height=350, background_color=None, mode="RGBA",
                                   colormap="cool" if sent == "Positif" else ("autumn" if sent == "Negatif" else "Set2"),
                                   max_words=80, contour_width=1, contour_color=SENTIMENT_COLORS[sent]).generate(texts)
                    fig_wc, ax_wc = plt.subplots(figsize=(6, 3.5))
                    ax_wc.imshow(wc, interpolation="bilinear")
                    ax_wc.axis("off")
                    fig_wc.patch.set_alpha(0)
                    st.pyplot(fig_wc)
                    plt.close(fig_wc)
                else:
                    st.caption(t["no_data"])


# ── TAB 3 — MODEL & EVALUATION ──────────────────────────────────────────────
with tab3:
    result = st.session_state.model_result
    if result is None:
        st.info(t["train_first_model"])
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", f"{result['accuracy']*100:.2f}%")
        m2.metric("Precision", f"{result['precision']*100:.2f}%")
        m3.metric("Recall", f"{result['recall']*100:.2f}%")
        m4.metric("F1-Score", f"{result['f1_score']*100:.2f}%")

        st.markdown("---")

        # Export model PDF
        pdf_bytes = export_model_pdf(result)
        st.download_button(t["export_model_pdf"], pdf_bytes, "model_report.pdf", "application/pdf")

        st.markdown("---")

        ev_col1, ev_col2 = st.columns(2)
        with ev_col1:
            st.markdown(f"##### {t['confusion_matrix']}")
            cm = result["confusion_matrix"]
            labels = result["labels"]
            fig_cm = px.imshow(cm, x=labels, y=labels, text_auto=True, color_continuous_scale="Purples",
                               labels={"x": t["prediction_label"], "y": t["actual_label"], "color": t["count_label"]})
            fig_cm.update_layout(margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig_cm, use_container_width=True)

        with ev_col2:
            st.markdown(f"##### {t['classification_report']}")
            report = result["classification_report"]
            report_df = pd.DataFrame(report).transpose()
            keep_rows = [r for r in report_df.index if r in labels or r == "weighted avg"]
            report_df = report_df.loc[keep_rows]
            for c in ["precision", "recall", "f1-score"]:
                if c in report_df.columns:
                    report_df[c] = report_df[c].apply(lambda v: f"{v*100:.2f}%")
            if "support" in report_df.columns:
                report_df["support"] = report_df["support"].astype(int)
            st.dataframe(report_df, use_container_width=True)

            st.markdown("---")
            st.markdown(f"**{t['ngram_label']}:** `{ngram_choice}`")
            st.markdown(f"**{t['train_size_label']}:** {result['train_size']}  |  **{t['test_size_label']}:** {result['test_size']}")
            st.markdown(f"**{t['total_features']}:** {len(result['feature_names']):,}")
            if "best_params" in result:
                st.markdown(f"**Best Alpha:** `{result['best_params'].get('alpha', 'N/A')}`")

        st.markdown("---")
        st.markdown(f"##### {t['ngram_comparison']}")
        comp = st.session_state.ngram_comparison
        if comp is not None:
            st.dataframe(comp, use_container_width=True)
            fig_comp = px.bar(
                comp.melt(id_vars=["N-Gram"], value_vars=["Accuracy", "Precision", "Recall", "F1-Score"]),
                x="N-Gram", y="value", color="variable", barmode="group",
                labels={"value": t["score_label"], "variable": t["metric_label"]},
                color_discrete_sequence=["#667eea", "#764ba2", "#f093fb", "#10b981"],
            )
            fig_comp.update_layout(margin=dict(l=20, r=20, t=30, b=40),
                                   legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5))
            st.plotly_chart(fig_comp, use_container_width=True)
        else:
            st.caption(t["click_compare"])


# ── TAB 4 — PREDICTION ──────────────────────────────────────────────────────
with tab4:
    result = st.session_state.model_result
    if result is None:
        st.info(t["train_first_model"])
    else:
        st.markdown(f"##### {t['predict_title']}")
        user_text = st.text_area(t["predict_input"], placeholder=t["predict_placeholder"], height=120)

        if st.button(t["predict_btn"], use_container_width=True, type="primary"):
            if user_text.strip():
                pred, proba, cleaned = predict_raw_text(
                    user_text, result["model"], result["vectorizer"],
                    use_stemming=st.session_state.use_stemming,
                )
                st.markdown("---")

                badge_class = f"badge-{pred.lower()}"
                st.markdown(f"""
                <div style="text-align:center; padding:24px 0;">
                    <p style="font-size:1rem; opacity:.7; margin-bottom:8px;">{t['predict_result']}</p>
                    <span class="badge {badge_class}" style="font-size:1.5rem; padding:10px 32px;">{pred}</span>
                </div>
                """, unsafe_allow_html=True)

                st.markdown(f"##### {t['prob_per_class']}")
                prob_cols = st.columns(len(proba))
                for i, (cls, pct) in enumerate(sorted(proba.items())):
                    with prob_cols[i]:
                        st.metric(cls, f"{pct:.1f}%")
                        st.progress(pct / 100)

                with st.expander(t["detail_preprocess"]):
                    st.markdown(f"**{t['original_text']}:** {user_text}")
                    st.markdown(f"**{t['after_preprocess_text']}:** `{cleaned}`")
            else:
                st.warning(t["enter_text_first"])

        st.markdown("---")
        st.markdown(f"##### {t['example_texts']}")
        examples = [
            "Game terbaik! Grafisnya keren banget dan gameplay seru abis!",
            "Lumayan buat ngisi waktu, tapi agak lag di HP kentang",
            "Sampah! Penuh cheater dan bug! Uninstall!",
        ]
        for ex in examples:
            if st.button(f"📝 {ex[:60]}...", key=f"ex_{hash(ex)}"):
                pred, proba, cleaned = predict_raw_text(
                    ex, result["model"], result["vectorizer"],
                    use_stemming=st.session_state.use_stemming,
                )
                badge_class = f"badge-{pred.lower()}"
                st.markdown(f'<span class="badge {badge_class}">{pred}</span> — {ex}', unsafe_allow_html=True)
                st.json(proba)
