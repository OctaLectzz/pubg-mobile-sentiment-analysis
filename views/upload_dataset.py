"""Upload Dataset Page — data source management."""
import streamlit as st
import pandas as pd
import os

from scraper import (
    generate_sample_data, scrape_reviews, scrape_multiple_countries,
    OUTPUT_FILE, COUNTRY_OPTIONS,
)


def render(t):
    """Render the Upload Dataset page."""
    st.markdown(f"### 📂 {t.get('nav_upload', 'Upload Dataset')}")
    st.markdown(f"_{t.get('upload_desc', 'Pilih sumber data untuk memulai analisis sentimen.')}_")

    # ── Data Source Tabs ──
    tab1, tab2, tab3 = st.tabs([
        f"📦 {t.get('data_source_sample', 'Data Sample')}",
        f"🌐 {t.get('data_source_scrape', 'Scrape App Store')}",
        f"📄 {t.get('data_source_upload', 'Upload CSV')}",
    ])

    # ── Tab 1: Sample Data ──
    with tab1:
        st.markdown(f"""
        <div class="glass-card" style="padding:24px;">
            <h4>📦 {t.get('sample_title', 'Data Sample PUBG Mobile')}</h4>
            <p style="opacity:.8;">{t.get('sample_desc', 'Gunakan data sample berisi 100 review PUBG Mobile dalam Bahasa Indonesia untuk demo dan pengujian cepat.')}</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("")

        if st.button(t["load_sample"], use_container_width=True, type="primary"):
            with st.spinner(t["load_sample_spinner"]):
                generate_sample_data()
            st.session_state.df_raw = pd.read_csv(OUTPUT_FILE)
            st.session_state.df_processed = None
            st.session_state.model_result = None
            st.session_state.ngram_comparison = None
            st.success(f"✅ {len(st.session_state.df_raw)} {t['load_sample_success']}")
            st.rerun()

        # Auto-load existing file
        if st.session_state.df_raw is None and os.path.exists(OUTPUT_FILE):
            st.session_state.df_raw = pd.read_csv(OUTPUT_FILE)

    # ── Tab 2: Scrape App Store ──
    with tab2:
        st.markdown(f"""
        <div class="glass-card" style="padding:24px;">
            <h4>🌐 {t.get('scrape_title', 'Scrape Review dari Apple App Store')}</h4>
            <p style="opacity:.8;">{t.get('scrape_desc', 'Ambil data review secara real-time dari App Store berbagai negara menggunakan RSS Feed.')}</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("")

        select_all = st.checkbox(t["select_all"], value=False)
        country_names = list(COUNTRY_OPTIONS.keys())
        if select_all:
            selected_countries = st.multiselect(t["select_countries"], country_names, default=country_names)
        else:
            selected_countries = st.multiselect(t["select_countries"], country_names, default=["🇮🇩 Indonesia"])

        scrape_count = st.number_input(t["review_count"], min_value=50, max_value=2000, value=200, step=50)

        if st.button(t["scrape_btn"], use_container_width=True, type="primary"):
            if selected_countries:
                country_codes = [COUNTRY_OPTIONS[c] for c in selected_countries]
                country_str = ", ".join(selected_countries[:3]) + ("..." if len(selected_countries) > 3 else "")
                msg = t["scrape_spinner"].format(count=scrape_count, countries=country_str)
                with st.spinner(msg):
                    if len(country_codes) == 1:
                        df_scraped = scrape_reviews(country=country_codes[0], how_many=scrape_count)
                    else:
                        per_country = max(50, scrape_count // len(country_codes) + 10)
                        df_scraped = scrape_multiple_countries(country_codes, how_many_per_country=per_country)
                    if df_scraped is not None and not df_scraped.empty and len(df_scraped) > scrape_count:
                        df_scraped = df_scraped.sample(n=scrape_count, random_state=42).reset_index(drop=True)
                if df_scraped is not None and not df_scraped.empty:
                    st.session_state.df_raw = df_scraped
                    st.session_state.df_processed = None
                    st.session_state.model_result = None
                    st.session_state.ngram_comparison = None
                    st.success(f"✅ {len(df_scraped)} {t['scrape_success']}")
                    st.rerun()
                else:
                    st.error(t["scrape_fail"])

    # ── Tab 3: Upload CSV ──
    with tab3:
        st.markdown(f"""
        <div class="glass-card" style="padding:24px;">
            <h4>📄 {t.get('csv_title', 'Upload File CSV')}</h4>
            <p style="opacity:.8;">{t.get('csv_desc', 'Upload file CSV Anda sendiri. Kolom utama yang dibutuhkan: <code>review</code> (atau <code>review_text</code>) dan <code>rating</code>. Kolom tambahan lainnya seperti <code>title</code>, <code>userName</code>, <code>date</code>, dan <code>country</code> akan otomatis diimpor jika tersedia.')}</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("")

        uploaded = st.file_uploader(t["upload_csv"], type=["csv"])
        if uploaded is not None:
            df_uploaded = pd.read_csv(uploaded)
            
            # Map columns to internal names
            rename_map = {}
            if "review" in df_uploaded.columns and "review_text" not in df_uploaded.columns:
                rename_map["review"] = "review_text"
            if "userName" in df_uploaded.columns and "username" not in df_uploaded.columns:
                rename_map["username"] = "username"
            if "userName" in df_uploaded.columns and "username" not in df_uploaded.columns and "userName" not in rename_map:
                rename_map["userName"] = "username"
                
            if rename_map:
                df_uploaded = df_uploaded.rename(columns=rename_map)
                
            # If rating column exists but sentiment doesn't, auto-generate sentiment
            if "rating" in df_uploaded.columns and "sentiment" not in df_uploaded.columns:
                from scraper import label_sentiment
                df_uploaded["sentiment"] = df_uploaded["rating"].apply(label_sentiment)
                
            # Ensure date column has correct string format if it exists
            if "date" in df_uploaded.columns:
                df_uploaded["date"] = df_uploaded["date"].astype(str).str[:10]
                
            st.session_state.df_raw = df_uploaded
            st.session_state.df_processed = None
            st.session_state.model_result = None
            st.session_state.ngram_comparison = None
            st.success(f"✅ {len(st.session_state.df_raw)} {t['rows_loaded']}")

    # ── Data Preview ──
    st.markdown("---")
    df = st.session_state.get("df_raw")
    if df is not None and len(df) > 0:
        st.markdown(f"##### 📋 {t.get('data_preview', 'Preview Data yang Dimuat')}")

        # Version filter
        if "version" in df.columns:
            versions = sorted(df["version"].dropna().unique().tolist())
            selected_versions = st.multiselect(t["filter_version"], options=versions, default=versions)
            if selected_versions:
                df_filtered = df[df["version"].isin(selected_versions)]
            else:
                df_filtered = df
        else:
            df_filtered = df

        col1, col2, col3 = st.columns(3)
        col1.metric(t["total_reviews"], f"{len(df_filtered):,}")
        if "rating" in df.columns:
            col2.metric(t.get("avg_rating", "Rata-rata Rating"), f"{df_filtered['rating'].mean():.2f} ⭐")
        sentiment_counts = df_filtered["sentiment"].value_counts() if "sentiment" in df_filtered.columns else {}
        if len(sentiment_counts) > 0:
            dominant = sentiment_counts.index[0]
            col3.metric(t.get("dominant_sentiment", "Sentimen Dominan"), dominant)

        display_cols = [c for c in ["review_text", "rating", "sentiment", "version", "date", "country"] if c in df_filtered.columns]
        st.dataframe(df_filtered[display_cols], use_container_width=True, height=400)
    else:
        st.info(t.get("no_data_yet", "📭 Belum ada data dimuat. Pilih salah satu sumber data di atas."))
