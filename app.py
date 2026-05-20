"""
Streamlit Dashboard — PUBG Mobile Sentiment Analysis
Naïve Bayes Classifier with N-Gram Feature Extraction

Main router with sidebar navigation and modern CSS theme.
"""
import streamlit as st
import os

from config.languages import get_lang, LANGUAGES
from model import NGRAM_OPTIONS

# View modules (named 'views' instead of 'pages' to avoid Streamlit auto-detection)
from views import (
    overview, upload_dataset, data_exploration,
    preprocessing_page, ngram_extraction, train_model, classification,
    evaluation, visualization, wordcloud_page,
    prediction, recommendations, about,
)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="PUBG Mobile Sentiment Analysis",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ══════════════════════════════════════════════════════════════════════════════
#  CSS THEME — Modern Glassmorphism Design
# ══════════════════════════════════════════════════════════════════════════════
def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* ── Metric Cards ── */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(102,126,234,.08), rgba(118,75,162,.06));
        border: 1px solid rgba(128,128,128,.12);
        border-radius: 16px;
        padding: 20px 24px;
        box-shadow: 0 4px 16px rgba(0,0,0,.06);
        transition: all .25s ease;
        position: relative;
        overflow: hidden;
    }
    div[data-testid="stMetric"]::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, #667eea, #764ba2, #f093fb);
        border-radius: 16px 16px 0 0;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 24px rgba(102,126,234,.15);
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 700;
    }

    /* ── Tabs ── */
    button[data-baseweb="tab"] {
        font-weight: 600 !important;
        font-size: .95rem !important;
        border-radius: 10px 10px 0 0 !important;
        transition: all .2s ease;
    }

    /* ── Tables ── */
    .stDataFrame { border-radius: 12px; overflow: hidden; }

    /* ── Hero Banner ── */
    .hero-banner {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        border-radius: 18px;
        padding: 34px 38px;
        margin-bottom: 26px;
        color: white;
        box-shadow: 0 8px 32px rgba(102,126,234,.35);
        animation: heroGlow 4s ease-in-out infinite alternate;
        position: relative;
        overflow: hidden;
    }
    .hero-banner::after {
        content: '';
        position: absolute;
        top: -50%; right: -30%;
        width: 60%; height: 200%;
        background: radial-gradient(ellipse, rgba(255,255,255,.08), transparent 70%);
        pointer-events: none;
    }
    @keyframes heroGlow {
        0% { box-shadow: 0 8px 32px rgba(102,126,234,.35); }
        100% { box-shadow: 0 8px 40px rgba(118,75,162,.45); }
    }
    .hero-banner h1 { margin: 0 0 6px 0; font-size: 2rem; font-weight: 800; }
    .hero-banner h2 { margin: 0 0 6px 0; font-size: 1.6rem; font-weight: 700; color: white; }
    .hero-banner p  { margin: 0; opacity: .88; font-size: 1.05rem; }

    /* ── Glass Cards ── */
    .glass-card {
        background: linear-gradient(135deg, rgba(102,126,234,.06), rgba(118,75,162,.04));
        border: 1px solid rgba(128,128,128,.1);
        border-radius: 14px;
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        transition: all .25s ease;
    }
    .glass-card:hover {
        border-color: rgba(102,126,234,.25);
        box-shadow: 0 6px 20px rgba(102,126,234,.1);
        transform: translateY(-2px);
    }

    /* ── Step Badge ── */
    .step-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 28px; height: 28px;
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border-radius: 50%;
        font-weight: 700;
        font-size: .8rem;
        flex-shrink: 0;
    }

    /* ── Sentiment Badges ── */
    .badge {
        display: inline-block;
        padding: 5px 16px;
        border-radius: 20px;
        font-weight: 600;
        font-size: .82rem;
        letter-spacing: .3px;
    }
    .badge-positif { background: linear-gradient(135deg, #10b981, #059669); color: #fff; }
    .badge-netral  { background: linear-gradient(135deg, #f59e0b, #d97706); color: #fff; }
    .badge-negatif { background: linear-gradient(135deg, #ef4444, #dc2626); color: #fff; }

    /* ── Sidebar Styling ── */
    section[data-testid="stSidebar"] > div:first-child {
        padding-top: .5rem;
    }
    section[data-testid="stSidebar"] .stRadio > label {
        font-weight: 600;
        font-size: .85rem;
        letter-spacing: .5px;
        color: rgba(128,128,128,.7);
        text-transform: uppercase;
        margin-top: 6px;
    }
    section[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label {
        padding: 8px 14px !important;
        border-radius: 10px !important;
        font-weight: 500 !important;
        font-size: .88rem !important;
        transition: all .2s ease !important;
        text-transform: none !important;
        letter-spacing: normal !important;
        cursor: pointer;
    }
    section[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label:hover {
        background: rgba(102,126,234,.08) !important;
    }
    section[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label[data-checked="true"] {
        background: linear-gradient(135deg, rgba(102,126,234,.15), rgba(118,75,162,.1)) !important;
        border-left: 3px solid #667eea !important;
    }

    /* ── Sidebar Brand ── */
    .sidebar-brand {
        text-align: center;
        padding: 12px 0 8px;
    }
    .sidebar-brand h2 {
        margin: 0;
        font-size: 1.3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea, #764ba2, #f093fb);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .sidebar-brand p {
        margin: 2px 0 0;
        font-size: .78rem;
        opacity: .6;
    }

    /* ── Category Headers via CSS ── */
    .nav-category {
        font-size: .72rem;
        font-weight: 700;
        letter-spacing: 1.2px;
        text-transform: uppercase;
        opacity: .45;
        padding: 10px 0 4px 4px;
        margin-top: 4px;
    }

    /* Category separators in single radio nav */
    div[data-testid="stSidebar"] div.nav-radio > div[role="radiogroup"] > label:nth-child(5),
    div[data-testid="stSidebar"] div.nav-radio > div[role="radiogroup"] > label:nth-child(8),
    div[data-testid="stSidebar"] div.nav-radio > div[role="radiogroup"] > label:nth-child(13) {
        margin-top: 18px;
        position: relative;
    }
    div[data-testid="stSidebar"] div.nav-radio > div[role="radiogroup"] > label:nth-child(5)::before,
    div[data-testid="stSidebar"] div.nav-radio > div[role="radiogroup"] > label:nth-child(8)::before,
    div[data-testid="stSidebar"] div.nav-radio > div[role="radiogroup"] > label:nth-child(13)::before {
        content: '';
        position: absolute;
        top: -10px; left: 0; right: 0;
        height: 1px;
        background: rgba(128,128,128,.15);
    }

    /* ── Status Pills ── */
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: .75rem;
        font-weight: 600;
    }
    .status-active { background: rgba(16,185,129,.15); color: #10b981; }
    .status-inactive { background: rgba(239,68,68,.1); color: #ef4444; }

    /* ── Full-screen loading overlay ── */
    .loading-overlay {
        position: fixed; top: 0; left: 0; width: 100vw; height: 100vh;
        background: rgba(0,0,0,.55); backdrop-filter: blur(4px);
        z-index: 9999999;
        display: flex; align-items: center; justify-content: center;
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

    /* ── Smooth page entrance ── */
    .main .block-container {
        animation: fadeIn .3s ease-out;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(8px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* ── Buttons ── */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #667eea, #764ba2) !important;
        border: none !important;
        font-weight: 600 !important;
        border-radius: 10px !important;
        transition: all .2s ease !important;
    }
    .stButton > button[kind="primary"]:hover {
        box-shadow: 0 4px 16px rgba(102,126,234,.35) !important;
        transform: translateY(-1px) !important;
    }

    /* ── Download buttons ── */
    .stDownloadButton > button {
        border-radius: 10px !important;
        font-weight: 600 !important;
        transition: all .2s ease !important;
    }

    /* ── Expander ── */
    .streamlit-expanderHeader {
        font-weight: 600 !important;
        border-radius: 10px !important;
    }

    /* ── Progress bars ── */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2) !important;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)


inject_css()


# ══════════════════════════════════════════════════════════════════════════════
#  SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
def init_session_state():
    defaults = {
        "df_raw": None,
        "df_processed": None,
        "model_result": None,
        "ngram_comparison": None,
        "lang": "id",
        "use_stemming": True,
        "current_page": "🏠 Overview",
        "prediction_history": [],
        "ngram_choice": "Unigram+Bigram (1,2)",
        "test_size": 0.2,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_session_state()


# ══════════════════════════════════════════════════════════════════════════════
#  NAVIGATION CONFIG
# ══════════════════════════════════════════════════════════════════════════════
def get_menu_items(t):
    """Return ordered menu items grouped by category."""
    return {
        f"📊 {t['nav_cat_data']}": [
            f"🏠 {t['nav_overview']}",
            f"📂 {t['nav_upload']}",
            f"🔍 {t['nav_exploration']}",
            f"🚀 {t['nav_train']}",
        ],
        f"⚙️ {t['nav_cat_analysis']}": [
            f"🔧 {t['nav_preprocessing']}",
            f"📐 {t['nav_ngram']}",
            f"🧠 {t['nav_classification']}",
        ],
        f"📈 {t['nav_cat_results']}": [
            f"📋 {t['nav_evaluation']}",
            f"📊 {t['nav_visualization']}",
            f"☁️ {t['nav_wordcloud']}",
            f"🔮 {t['nav_prediction']}",
            f"💡 {t['nav_recommendations']}",
        ],
        f"ℹ️ {t['nav_cat_info']}": [
            f"📖 {t['nav_about']}",
        ],
    }


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
t = get_lang(st.session_state.lang)

with st.sidebar:
    # ── Brand ──
    st.markdown("""
    <div class="sidebar-brand">
        <h2>🎮 PUBG Mobile</h2>
        <p>Sentiment Analysis Dashboard</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Language Selector ──
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

    # Refresh translations after language change
    t = get_lang(st.session_state.lang)

    st.markdown("---")

    # ── Navigation ──
    menu_items = get_menu_items(t)
    all_pages = []
    for category, pages in menu_items.items():
        all_pages.extend(pages)

    # Ensure current_page is valid
    if st.session_state.current_page not in all_pages:
        st.session_state.current_page = all_pages[0]

    # Single radio for all pages — CSS handles visual category separators
    st.markdown(f'<div class="nav-category">📊 {t["nav_cat_data"]}</div>', unsafe_allow_html=True)
    with st.container():
        selected_page = st.radio(
            "Navigation",
            all_pages,
            index=all_pages.index(st.session_state.current_page),
            key="main_nav",
            label_visibility="collapsed",
        )
    if selected_page != st.session_state.current_page:
        st.session_state.current_page = selected_page
        st.rerun()

    st.markdown("---")

    # ── Data Status ──
    df = st.session_state.get("df_raw")
    model = st.session_state.get("model_result")

    if df is not None and len(df) > 0:
        st.markdown(f'<span class="status-pill status-active">● {t["data_loaded"]} ({len(df):,})</span>', unsafe_allow_html=True)
    else:
        st.markdown(f'<span class="status-pill status-inactive">○ {t["data_not_loaded"]}</span>', unsafe_allow_html=True)

    if model:
        st.markdown(f'<span class="status-pill status-active">● Model OK ({model["accuracy"]*100:.0f}%)</span>', unsafe_allow_html=True)
    else:
        st.markdown(f'<span class="status-pill status-inactive">○ Model —</span>', unsafe_allow_html=True)

    st.markdown("")

    st.markdown("---")
    st.caption(t["copyright"])


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE ROUTER
# ══════════════════════════════════════════════════════════════════════════════
# Build dispatch table: maps full page label → module
_dispatch = {}
_module_order = [
    overview, upload_dataset, data_exploration, train_model,
    preprocessing_page, ngram_extraction, classification,
    evaluation, visualization, wordcloud_page,
    prediction, recommendations, about,
]
for page_label, module in zip(all_pages, _module_order):
    _dispatch[page_label] = module

current = st.session_state.current_page
page_module = _dispatch.get(current, overview)
page_module.render(t)

