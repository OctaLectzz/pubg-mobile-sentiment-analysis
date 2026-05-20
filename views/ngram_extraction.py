"""N-Gram Extraction Page — feature analysis and top N-Gram display."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from model import NGRAM_OPTIONS


def render(t):
    """Render the N-Gram Extraction page."""
    st.markdown(f"### 📐 {t.get('nav_ngram', 'Ekstraksi N-Gram')}")
    st.markdown(f"_{t.get('ngram_page_desc', 'Analisis fitur N-Gram yang diekstrak dari teks review.')}_")

    # ── N-Gram Explanation ──
    st.markdown(f"##### 📚 {t.get('ngram_what', 'Apa itu N-Gram?')}")

    cols = st.columns(3)
    ngram_info = [
        ("Unigram (1,1)", t.get("unigram_desc", "Satu kata tunggal sebagai fitur."), "game, bagus, seru"),
        ("Bigram (2,2)", t.get("bigram_desc", "Pasangan dua kata berurutan."), "game bagus, sangat seru"),
        ("Trigram (3,3)", t.get("trigram_desc", "Kombinasi tiga kata berurutan."), "game sangat bagus"),
    ]
    for i, (title, desc, example) in enumerate(ngram_info):
        with cols[i]:
            st.markdown(f"""
            <div class="glass-card" style="padding:20px; text-align:center;">
                <h4 style="margin:0 0 8px; color:#667eea;">{title}</h4>
                <p style="opacity:.7; font-size:.85rem; margin:0 0 10px;">{desc}</p>
                <code style="font-size:.8rem;">{example}</code>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("")
    st.info(t.get("ngram_tfidf_note", "💡 Fitur N-Gram diekstrak menggunakan **TF-IDF Vectorizer** yang mengukur kepentingan setiap N-Gram dalam dokumen relatif terhadap seluruh korpus."))

    st.markdown("---")

    # ── Feature Analysis (requires trained model) ──
    result = st.session_state.get("model_result")
    if result is None:
        from views import redirect_to_train
        redirect_to_train(t)
        return

    # Model feature info
    st.markdown(f"##### 📊 {t.get('ngram_feature_info', 'Informasi Fitur')}")
    fi1, fi2, fi3 = st.columns(3)
    fi1.metric(t.get("ngram_range_label", "N-Gram Range"), str(result["ngram_range"]))
    fi2.metric(t.get("total_features", "Total Fitur"), f"{len(result['feature_names']):,}")
    fi3.metric(t.get("best_alpha", "Best Alpha"), f"{result.get('best_params', {}).get('alpha', 'N/A')}")

    st.markdown("---")

    # ── Top N-Gram Features ──
    st.markdown(f"##### 🏆 {t.get('top_features', 'Top N-Gram Features')}")

    vectorizer = result["vectorizer"]
    model = result["model"]
    feature_names = result["feature_names"]
    labels = result["labels"]

    # Get feature importance from model coefficients
    top_n = st.slider(t.get("top_n_slider", "Jumlah fitur teratas"), 10, 30, 15)

    tabs = st.tabs(labels)
    for tab, label in zip(tabs, labels):
        with tab:
            label_idx = list(model.classes_).index(label)
            # For ComplementNB, feature_log_prob_ gives log probabilities
            if hasattr(model, "feature_log_prob_"):
                scores = model.feature_log_prob_[label_idx]
            else:
                scores = np.zeros(len(feature_names))

            top_indices = np.argsort(scores)[-top_n:][::-1]
            top_features = [feature_names[i] for i in top_indices]
            top_scores = [float(scores[i]) for i in top_indices]

            df_top = pd.DataFrame({"Feature": top_features, "Score": top_scores})
            fig = px.bar(
                df_top, x="Score", y="Feature", orientation="h",
                color="Score", color_continuous_scale="Purples",
            )
            fig.update_layout(
                yaxis=dict(autorange="reversed"),
                margin=dict(l=20, r=20, t=10, b=20),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                showlegend=False, height=max(350, top_n * 25),
            )
            fig.update_coloraxes(showscale=False)
            st.plotly_chart(fig, use_container_width=True)

    # ── N-Gram Distribution ──
    st.markdown("---")
    st.markdown(f"##### 📈 {t.get('ngram_distribution', 'Distribusi Panjang N-Gram')}")
    word_counts = [len(f.split()) for f in feature_names]
    ngram_dist = pd.Series(word_counts).value_counts().sort_index()
    ngram_labels = {1: "Unigram", 2: "Bigram", 3: "Trigram"}

    dist_df = pd.DataFrame({
        "Tipe": [ngram_labels.get(k, f"{k}-gram") for k in ngram_dist.index],
        "Jumlah": ngram_dist.values,
    })
    fig_dist = px.bar(
        dist_df, x="Tipe", y="Jumlah",
        color="Tipe", color_discrete_sequence=["#667eea", "#764ba2", "#f093fb"],
    )
    fig_dist.update_layout(
        showlegend=False, margin=dict(l=20, r=20, t=30, b=20),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_dist, use_container_width=True)
