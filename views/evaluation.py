"""Model Evaluation Page — metrics, confusion matrix, classification report."""
import streamlit as st
import pandas as pd
import plotly.express as px
from utils.export import export_model_pdf


def render(t):
    """Render the Model Evaluation page."""
    st.markdown(f"### 📋 {t.get('nav_evaluation', 'Evaluasi Model')}")

    result = st.session_state.get("model_result")
    if result is None:
        from views import redirect_to_train
        redirect_to_train(t)
        return

    # ── Performance Metrics ──
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{result['accuracy']*100:.2f}%")
    m2.metric("Precision", f"{result['precision']*100:.2f}%")
    m3.metric("Recall", f"{result['recall']*100:.2f}%")
    m4.metric("F1-Score", f"{result['f1_score']*100:.2f}%")

    st.markdown("---")

    # ── Export ──
    pdf_bytes = export_model_pdf(result)
    st.download_button(
        t["export_model_pdf"], pdf_bytes, "model_report.pdf", "application/pdf",
        use_container_width=True,
    )

    st.markdown("---")

    # ── Confusion Matrix & Classification Report ──
    ev_col1, ev_col2 = st.columns(2)

    with ev_col1:
        st.markdown(f"##### {t['confusion_matrix']}")
        cm = result["confusion_matrix"]
        labels = result["labels"]
        fig_cm = px.imshow(
            cm, x=labels, y=labels, text_auto=True,
            color_continuous_scale="Purples",
            labels={"x": t["prediction_label"], "y": t["actual_label"], "color": t["count_label"]},
        )
        fig_cm.update_layout(
            margin=dict(l=20, r=20, t=30, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    with ev_col2:
        st.markdown(f"##### {t['classification_report']}")
        report = result["classification_report"]
        report_df = pd.DataFrame(report).transpose()
        labels = result["labels"]
        keep_rows = [r for r in report_df.index if r in labels or r == "weighted avg"]
        report_df = report_df.loc[keep_rows]
        for c in ["precision", "recall", "f1-score"]:
            if c in report_df.columns:
                report_df[c] = report_df[c].apply(lambda v: f"{v*100:.2f}%")
        if "support" in report_df.columns:
            report_df["support"] = report_df["support"].astype(int)
        st.dataframe(report_df, use_container_width=True)

    st.markdown("---")

    # ── Per-Class Performance Bar Chart ──
    st.markdown(f"##### 📊 {t.get('per_class_perf', 'Performa per Kelas')}")
    report_data = result["classification_report"]
    perf_rows = []
    for label in labels:
        if label in report_data:
            perf_rows.append({
                "Kelas": label,
                "Precision": round(report_data[label]["precision"] * 100, 2),
                "Recall": round(report_data[label]["recall"] * 100, 2),
                "F1-Score": round(report_data[label]["f1-score"] * 100, 2),
            })
    perf_df = pd.DataFrame(perf_rows)
    fig_perf = px.bar(
        perf_df.melt(id_vars=["Kelas"], value_vars=["Precision", "Recall", "F1-Score"]),
        x="Kelas", y="value", color="variable", barmode="group",
        labels={"value": "Skor (%)", "variable": "Metrik"},
        color_discrete_sequence=["#667eea", "#764ba2", "#f093fb"],
    )
    fig_perf.update_layout(
        margin=dict(l=20, r=20, t=30, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_perf, use_container_width=True)

    # ── Model Info ──
    st.markdown("---")
    st.markdown(f"##### ℹ️ {t.get('model_info', 'Informasi Model')}")
    info1, info2 = st.columns(2)
    with info1:
        ngram_choice = st.session_state.get("ngram_choice", "N/A")
        st.markdown(f"**{t['ngram_label']}:** `{ngram_choice}`")
        st.markdown(f"**{t['train_size_label']}:** {result['train_size']}  |  **{t['test_size_label']}:** {result['test_size']}")
    with info2:
        st.markdown(f"**{t['total_features']}:** {len(result['feature_names']):,}")
        if "best_params" in result:
            st.markdown(f"**Best Alpha:** `{result['best_params'].get('alpha', 'N/A')}`")
        st.markdown(f"**Model:** Complement Naïve Bayes")
