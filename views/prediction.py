"""Prediction Page — manual sentiment prediction with history."""
import streamlit as st

from model import predict_raw_text


def render(t):
    """Render the Prediction page."""
    st.markdown(f"### 🔮 {t.get('nav_prediction', 'Prediksi')}")
    st.markdown(f"_{t.get('predict_page_desc', 'Masukkan teks review untuk memprediksi sentimennya secara real-time.')}_")

    result = st.session_state.get("model_result")
    if result is None:
        from views import redirect_to_train
        redirect_to_train(t)
        return

    # ── Init prediction history ──
    if "prediction_history" not in st.session_state:
        st.session_state.prediction_history = []

    # ── Prediction Input ──
    st.markdown(f"##### ✍️ {t['predict_title']}")
    user_text = st.text_area(
        t["predict_input"],
        placeholder=t["predict_placeholder"],
        height=120,
        key="predict_textarea",
    )

    char_count = len(user_text) if user_text else 0
    st.caption(f"📝 {char_count} karakter")

    if st.button(t["predict_btn"], use_container_width=True, type="primary"):
        if user_text.strip():
            pred, proba, cleaned = predict_raw_text(
                user_text, result["model"], result["vectorizer"],
                use_stemming=st.session_state.get("use_stemming", True),
            )

            # Save to history
            st.session_state.prediction_history.insert(0, {
                "text": user_text[:100] + ("..." if len(user_text) > 100 else ""),
                "prediction": pred,
                "proba": proba,
            })

            st.markdown("---")

            # ── Result Display ──
            badge_class = f"badge-{pred.lower()}"
            st.markdown(f"""
            <div style="text-align:center; padding:28px 0;">
                <p style="font-size:1rem; opacity:.7; margin-bottom:10px;">{t['predict_result']}</p>
                <span class="badge {badge_class}" style="font-size:1.6rem; padding:12px 36px;">{pred}</span>
            </div>
            """, unsafe_allow_html=True)

            # ── Probability Gauges ──
            st.markdown(f"##### {t['prob_per_class']}")
            prob_cols = st.columns(len(proba))
            for i, (cls, pct) in enumerate(sorted(proba.items())):
                with prob_cols[i]:
                    st.metric(cls, f"{pct:.1f}%")
                    st.progress(pct / 100)

            # ── Preprocessing Detail ──
            with st.expander(t["detail_preprocess"]):
                st.markdown(f"**{t['original_text']}:** {user_text}")
                st.markdown(f"**{t['after_preprocess_text']}:** `{cleaned}`")
        else:
            st.warning(t["enter_text_first"])

    # ── Example Texts ──
    st.markdown("---")
    st.markdown(f"##### {t['example_texts']}")

    examples = [
        ("😍", "Game terbaik! Grafisnya keren banget dan gameplay seru abis!"),
        ("😐", "Lumayan buat ngisi waktu, tapi agak lag di HP kentang"),
        ("😡", "Sampah! Penuh cheater dan bug! Uninstall!"),
        ("🤔", "Game biasa aja, kadang seru kadang kesel karena cheater"),
        ("🔥", "Update terbarunya mantap, mode baru seru banget!"),
    ]

    for emoji, ex in examples:
        if st.button(f"{emoji} {ex[:65]}{'...' if len(ex) > 65 else ''}", key=f"ex_{hash(ex)}", use_container_width=True):
            pred, proba, cleaned = predict_raw_text(
                ex, result["model"], result["vectorizer"],
                use_stemming=st.session_state.get("use_stemming", True),
            )
            badge_class = f"badge-{pred.lower()}"
            st.markdown(f'<span class="badge {badge_class}">{pred}</span> — {ex}', unsafe_allow_html=True)

            prob_cols = st.columns(len(proba))
            for i, (cls, pct) in enumerate(sorted(proba.items())):
                with prob_cols[i]:
                    st.metric(cls, f"{pct:.1f}%")

    # ── Prediction History ──
    history = st.session_state.get("prediction_history", [])
    if history:
        st.markdown("---")
        st.markdown(f"##### 📜 {t.get('predict_history', 'Riwayat Prediksi')}")
        for i, entry in enumerate(history[:10]):
            badge_class = f"badge-{entry['prediction'].lower()}"
            st.markdown(
                f'<span class="badge {badge_class}" style="font-size:.8rem;">{entry["prediction"]}</span> '
                f'<span style="opacity:.8; font-size:.9rem;">{entry["text"]}</span>',
                unsafe_allow_html=True,
            )
