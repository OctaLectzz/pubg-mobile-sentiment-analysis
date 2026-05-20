"""Word Cloud Page — dedicated word cloud visualization."""
import streamlit as st
import matplotlib.pyplot as plt
from wordcloud import WordCloud

SENTIMENT_COLORS = {"Positif": "#10b981", "Netral": "#f59e0b", "Negatif": "#ef4444"}
COLORMAPS = {
    "Positif": ["cool", "winter", "GnBu"],
    "Netral": ["Set2", "Pastel1", "Set3"],
    "Negatif": ["autumn", "YlOrRd", "Reds"],
}


def render(t):
    """Render the Word Cloud page."""
    st.markdown(f"### ☁️ {t.get('nav_wordcloud', 'Word Cloud')}")
    st.markdown(f"_{t.get('wc_desc', 'Visualisasi kata-kata yang paling sering muncul dalam setiap kategori sentimen.')}_")

    df_proc = st.session_state.get("df_processed")
    if df_proc is None:
        from views import redirect_to_train
        redirect_to_train(t)
        return

    # ── Controls ──
    st.markdown(f"##### ⚙️ {t.get('wc_settings', 'Pengaturan Word Cloud')}")
    ctrl1, ctrl2 = st.columns(2)
    with ctrl1:
        max_words = st.slider(t.get("wc_max_words", "Maksimum Kata"), 30, 200, 80, step=10)
    with ctrl2:
        bg_option = st.selectbox(
            t.get("wc_background", "Background"),
            [t.get("wc_transparent", "Transparan"), t.get("wc_white", "Putih"), t.get("wc_dark", "Gelap")],
        )
    bg_map = {
        t.get("wc_transparent", "Transparan"): None,
        t.get("wc_white", "Putih"): "white",
        t.get("wc_dark", "Gelap"): "#1a1a2e",
    }
    bg_color = bg_map.get(bg_option)
    mode = "RGBA" if bg_color is None else "RGB"

    st.markdown("---")

    # ── Word Clouds per Sentiment ──
    st.markdown(f"##### ☁️ {t['wordcloud_title']}")

    wc_cols = st.columns(3)
    for idx, sent in enumerate(["Positif", "Netral", "Negatif"]):
        texts = " ".join(df_proc[df_proc["sentiment"] == sent]["cleaned_text"].tolist())
        with wc_cols[idx]:
            # Sentiment badge header
            badge_class = f"badge-{sent.lower()}"
            st.markdown(f'<span class="badge {badge_class}" style="font-size:.95rem; padding:6px 18px;">{sent}</span>', unsafe_allow_html=True)
            st.markdown("")

            if texts.strip():
                colormap = COLORMAPS[sent][0]
                wc = WordCloud(
                    width=700, height=420,
                    background_color=bg_color, mode=mode,
                    colormap=colormap,
                    max_words=max_words,
                    contour_width=1,
                    contour_color=SENTIMENT_COLORS[sent],
                    prefer_horizontal=0.7,
                )
                wc.generate(texts)

                fig_wc, ax_wc = plt.subplots(figsize=(7, 4.2))
                ax_wc.imshow(wc, interpolation="bilinear")
                ax_wc.axis("off")
                if bg_color is None:
                    fig_wc.patch.set_alpha(0)
                else:
                    fig_wc.patch.set_facecolor(bg_color)
                st.pyplot(fig_wc)
                plt.close(fig_wc)

                # Word count stat
                word_count = len(texts.split())
                unique_words = len(set(texts.split()))
                st.caption(f"📊 {word_count:,} kata  |  {unique_words:,} unik")
            else:
                st.caption(t["no_data"])

    # ── Combined Word Cloud ──
    st.markdown("---")
    st.markdown(f"##### 🌐 {t.get('wc_combined', 'Word Cloud Gabungan (Semua Sentimen)')}")

    all_texts = " ".join(df_proc["cleaned_text"].tolist())
    if all_texts.strip():
        wc_all = WordCloud(
            width=1200, height=500,
            background_color=bg_color, mode=mode,
            colormap="viridis",
            max_words=max_words * 2,
            prefer_horizontal=0.7,
        )
        wc_all.generate(all_texts)

        fig_all, ax_all = plt.subplots(figsize=(12, 5))
        ax_all.imshow(wc_all, interpolation="bilinear")
        ax_all.axis("off")
        if bg_color is None:
            fig_all.patch.set_alpha(0)
        else:
            fig_all.patch.set_facecolor(bg_color)
        st.pyplot(fig_all)
        plt.close(fig_all)
    else:
        st.caption(t["no_data"])
