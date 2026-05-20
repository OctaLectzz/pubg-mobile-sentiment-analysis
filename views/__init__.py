# Views package for PUBG Mobile Sentiment Analysis Dashboard
import streamlit as st


def redirect_to_train(t):
    """Show info message and button to navigate to training page."""
    st.warning(t.get("train_first_model", "⬅️ Latih model terlebih dahulu."))
    st.markdown("")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button(
            f"🚀 {t.get('go_to_train', 'Pergi ke Latih Model')}",
            type="primary",
            use_container_width=True,
            key="redirect_train",
        ):
            st.session_state.current_page = f"🚀 {t.get('nav_train', 'Latih Model')}"
            st.rerun()


def redirect_to_upload(t):
    """Show info message and button to navigate to upload page."""
    st.info(t.get("load_data_first", "⬅️ Muat data terlebih dahulu."))
    st.markdown("")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button(
            f"📂 {t.get('go_to_upload', 'Pergi ke Upload Dataset')}",
            type="primary",
            use_container_width=True,
            key="redirect_upload",
        ):
            st.session_state.current_page = f"📂 {t.get('nav_upload', 'Upload Dataset')}"
            st.rerun()
