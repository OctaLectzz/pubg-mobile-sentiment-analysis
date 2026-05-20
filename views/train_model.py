"""Train Model Page — dedicated page for model training with config."""
import streamlit as st

from preprocessing import preprocess_dataframe
from model import build_model, compare_ngrams, save_model, NGRAM_OPTIONS


def render(t):
    """Render the Train Model page."""
    st.markdown(f"### 🚀 {t.get('nav_train', 'Latih Model')}")
    st.markdown(f"_{t.get('train_page_desc', 'Konfigurasi parameter dan latih model Naïve Bayes untuk klasifikasi sentimen.')}_")

    df = st.session_state.get("df_raw")
    if df is None or len(df) == 0:
        from views import redirect_to_upload
        redirect_to_upload(t)
        return

    # ── Model Configuration ──
    st.markdown(f"##### ⚙️ {t.get('model_config', 'Konfigurasi Model')}")

    cfg1, cfg2, cfg3 = st.columns(3)
    with cfg1:
        ngram_choice = st.selectbox(
            t.get("ngram_type", "Tipe N-Gram"),
            list(NGRAM_OPTIONS.keys()),
            index=list(NGRAM_OPTIONS.keys()).index(st.session_state.get("ngram_choice", "Unigram+Bigram (1,2)")),
            key="train_ngram_select",
        )
        st.session_state.ngram_choice = ngram_choice
    with cfg2:
        use_stemming = st.checkbox(
            t.get("use_stemming", "Gunakan Stemming"),
            value=st.session_state.get("use_stemming", True),
            key="train_stemming",
        )
        st.session_state.use_stemming = use_stemming
    with cfg3:
        test_pct = st.slider(
            t.get("test_size", "Test Size (%)"),
            10, 40,
            int(st.session_state.get("test_size", 0.2) * 100),
            5,
            key="train_test_slider",
        )
        st.session_state.test_size = test_pct / 100

    # ── Config Summary ──
    st.markdown("---")
    sum1, sum2, sum3, sum4 = st.columns(4)
    sum1.metric(t.get("total_reviews", "Total Review"), f"{len(df):,}")
    sum2.metric(t.get("ngram_label", "N-Gram"), ngram_choice.split(" ")[0])
    sum3.metric("Stemming", "✅ Aktif" if use_stemming else "❌ Off")
    sum4.metric(t.get("test_size", "Test Size"), f"{test_pct}%")

    st.markdown("---")

    # ── Training Actions ──
    train_col, compare_col = st.columns(2)

    with train_col:
        st.markdown(f"""
        <div class="glass-card" style="padding:22px;">
            <h4>🧠 {t.get('train_model_title', 'Latih Model Naïve Bayes')}</h4>
            <p style="opacity:.7; font-size:.85rem;">{t.get('train_model_desc', 'Melatih model Complement Naïve Bayes dengan GridSearchCV untuk menemukan hyperparameter optimal.')}</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("")
        if st.button(t["train_btn"], use_container_width=True, type="primary", key="train_main"):
            with st.spinner(t["preprocess_spinner"]):
                df_proc = preprocess_dataframe(df, use_stemming=use_stemming)
                st.session_state.df_processed = df_proc
            with st.spinner(t["train_spinner"]):
                ngram_range = NGRAM_OPTIONS[ngram_choice]
                result = build_model(
                    df_proc["cleaned_text"], df_proc["sentiment"],
                    ngram_range=ngram_range, test_size=st.session_state.test_size,
                )
                st.session_state.model_result = result
                save_model(result["model"], result["vectorizer"])
            st.toast(t["train_success"], icon="🎉")
            st.rerun()

    with compare_col:
        st.markdown(f"""
        <div class="glass-card" style="padding:22px;">
            <h4>📊 {t.get('compare_title', 'Bandingkan N-Gram')}</h4>
            <p style="opacity:.7; font-size:.85rem;">{t.get('compare_desc', 'Membandingkan performa semua konfigurasi N-Gram secara otomatis.')}</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("")
        if st.button(t["compare_btn"], use_container_width=True, type="secondary", key="compare_main"):
            with st.spinner(t["preprocess_spinner"]):
                df_proc = preprocess_dataframe(df, use_stemming=use_stemming)
                st.session_state.df_processed = df_proc
            with st.spinner(t["compare_spinner"]):
                comparison = compare_ngrams(
                    df_proc["cleaned_text"], df_proc["sentiment"],
                    test_size=st.session_state.test_size,
                )
                st.session_state.ngram_comparison = comparison
            st.toast(t["compare_success"], icon="📊")
            st.rerun()

    # ── Quick Results ──
    result = st.session_state.get("model_result")
    if result is not None:
        st.markdown("---")
        st.success(f"✅ {t.get('model_trained', 'Model berhasil dilatih!')}")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", f"{result['accuracy']*100:.2f}%")
        m2.metric("Precision", f"{result['precision']*100:.2f}%")
        m3.metric("Recall", f"{result['recall']*100:.2f}%")
        m4.metric("F1-Score", f"{result['f1_score']*100:.2f}%")

        st.info(t.get("train_done_hint", "💡 Lihat hasil detail di menu **📋 Evaluasi Model** atau coba prediksi di **🔮 Prediksi**."))
