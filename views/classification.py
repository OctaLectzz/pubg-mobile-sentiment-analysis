"""Classification Page — model training and N-Gram comparison."""
import streamlit as st
import pandas as pd
import plotly.express as px

from preprocessing import preprocess_dataframe
from model import build_model, compare_ngrams, save_model, NGRAM_OPTIONS

SENTIMENT_COLORS = {"Positif": "#10b981", "Netral": "#f59e0b", "Negatif": "#ef4444"}


def render(t):
    """Render the Classification / Training page."""
    st.markdown(f"### 🧠 {t.get('nav_classification', 'Klasifikasi Sentimen')}")
    st.markdown(f"_{t.get('classification_desc', 'Latih model Naïve Bayes dan bandingkan konfigurasi N-Gram.')}_")

    df = st.session_state.get("df_raw")
    if df is None or len(df) == 0:
        from views import redirect_to_upload
        redirect_to_upload(t)
        return

    # ── Current Configuration ──
    st.markdown(f"##### ⚙️ {t.get('current_config', 'Konfigurasi Saat Ini')}")

    ngram_choice = st.session_state.get("ngram_choice", "Unigram+Bigram (1,2)")
    test_size = st.session_state.get("test_size", 0.2)
    use_stemming = st.session_state.get("use_stemming", True)

    cfg1, cfg2, cfg3, cfg4 = st.columns(4)
    cfg1.metric(t.get("ngram_label", "N-Gram"), ngram_choice.split(" ")[0])
    cfg2.metric(t.get("test_size", "Test Size"), f"{int(test_size * 100)}%")
    cfg3.metric("Stemming", "✅ Aktif" if use_stemming else "❌ Off")
    cfg4.metric(t["total_reviews"], f"{len(df):,}")

    st.markdown("---")

    # ── Training Section ──
    st.markdown(f"##### 🚀 {t.get('train_section', 'Pelatihan Model')}")

    train_col1, train_col2 = st.columns(2)

    with train_col1:
        st.markdown(f"""
        <div class="glass-card" style="padding:20px;">
            <h4>🧠 {t.get('train_model_title', 'Latih Model Naïve Bayes')}</h4>
            <p style="opacity:.7; font-size:.85rem;">{t.get('train_model_desc', 'Melatih model Complement Naïve Bayes dengan GridSearchCV untuk hyperparameter tuning.')}</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("")
        if st.button(t["train_btn"], use_container_width=True, type="primary", key="btn_train_main"):
            with st.spinner(t["preprocess_spinner"]):
                df_proc = preprocess_dataframe(df, use_stemming=use_stemming)
                st.session_state.df_processed = df_proc
            with st.spinner(t["train_spinner"]):
                ngram_range = NGRAM_OPTIONS[ngram_choice]
                result = build_model(
                    df_proc["cleaned_text"], df_proc["sentiment"],
                    ngram_range=ngram_range, test_size=test_size,
                )
                st.session_state.model_result = result
                save_model(result["model"], result["vectorizer"])
            st.toast(t["train_success"], icon="🎉")
            st.rerun()

    with train_col2:
        st.markdown(f"""
        <div class="glass-card" style="padding:20px;">
            <h4>📊 {t.get('compare_title', 'Bandingkan N-Gram')}</h4>
            <p style="opacity:.7; font-size:.85rem;">{t.get('compare_desc', 'Membandingkan performa semua konfigurasi N-Gram untuk menemukan yang terbaik.')}</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("")
        if st.button(t["compare_btn"], use_container_width=True, type="secondary", key="btn_compare_main"):
            with st.spinner(t["preprocess_spinner"]):
                df_proc = preprocess_dataframe(df, use_stemming=use_stemming)
                st.session_state.df_processed = df_proc
            with st.spinner(t["compare_spinner"]):
                comparison = compare_ngrams(
                    df_proc["cleaned_text"], df_proc["sentiment"], test_size=test_size,
                )
                st.session_state.ngram_comparison = comparison
            st.toast(t["compare_success"], icon="📊")
            st.rerun()

    # ── Training Results ──
    result = st.session_state.get("model_result")
    if result is not None:
        st.markdown("---")
        st.markdown(f"##### ✅ {t.get('training_results', 'Hasil Pelatihan')}")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", f"{result['accuracy']*100:.2f}%")
        m2.metric("Precision", f"{result['precision']*100:.2f}%")
        m3.metric("Recall", f"{result['recall']*100:.2f}%")
        m4.metric("F1-Score", f"{result['f1_score']*100:.2f}%")

        info_col1, info_col2 = st.columns(2)
        with info_col1:
            st.markdown(f"**{t['ngram_label']}:** `{ngram_choice}`")
            st.markdown(f"**{t['train_size_label']}:** {result['train_size']}  |  **{t['test_size_label']}:** {result['test_size']}")
        with info_col2:
            st.markdown(f"**{t['total_features']}:** {len(result['feature_names']):,}")
            if "best_params" in result:
                st.markdown(f"**Best Alpha:** `{result['best_params'].get('alpha', 'N/A')}`")

    # ── N-Gram Comparison Results ──
    comp = st.session_state.get("ngram_comparison")
    if comp is not None:
        st.markdown("---")
        st.markdown(f"##### {t['ngram_comparison']}")
        st.dataframe(comp, use_container_width=True)

        fig_comp = px.bar(
            comp.melt(id_vars=["N-Gram"], value_vars=["Accuracy", "Precision", "Recall", "F1-Score"]),
            x="N-Gram", y="value", color="variable", barmode="group",
            labels={"value": t["score_label"], "variable": t["metric_label"]},
            color_discrete_sequence=["#667eea", "#764ba2", "#f093fb", "#10b981"],
        )
        fig_comp.update_layout(
            margin=dict(l=20, r=20, t=30, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_comp, use_container_width=True)

        # Highlight best
        best_idx = comp["F1-Score"].idxmax()
        best_ngram = comp.loc[best_idx, "N-Gram"]
        best_f1 = comp.loc[best_idx, "F1-Score"]
        st.success(f"🏆 **{t.get('best_ngram', 'N-Gram Terbaik')}:** {best_ngram} — F1-Score: {best_f1:.2f}%")
