"""
Export utilities for CSV and PDF generation.
"""
import io
import pandas as pd
from fpdf import FPDF


def export_csv(df):
    """Export DataFrame to CSV bytes for st.download_button."""
    buffer = io.BytesIO()
    df.to_csv(buffer, index=False, encoding="utf-8-sig")
    return buffer.getvalue()

def sanitize_text(text):
    """Sanitize text to be compatible with standard FPDF latin-1 fonts."""
    if not isinstance(text, str):
        text = str(text)
    # Replace common unicode equivalents
    text = text.replace('…', '...').replace('”', '"').replace('“', '"').replace("'", "'").replace("'", "'")
    # Encode to latin-1, ignoring non-supported characters (like emojis)
    return text.encode('latin-1', 'ignore').decode('latin-1')


class SentimentPDF(FPDF):
    """Custom PDF class for sentiment analysis reports."""

    def header(self):
        self.set_font("Helvetica", "B", 16)
        self.set_text_color(102, 126, 234)
        self.cell(0, 12, "PUBG Mobile Sentiment Analysis Report", align="C", new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(102, 126, 234)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(6)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def section_title(self, title):
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(50, 50, 80)
        self.ln(4)
        self.cell(0, 10, sanitize_text(title), new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(200, 200, 220)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(3)

    def key_value(self, key, value):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(80, 80, 100)
        self.cell(55, 7, sanitize_text(f"{key}:"), new_x="END")
        self.set_font("Helvetica", "", 10)
        self.set_text_color(30, 30, 30)
        self.cell(0, 7, sanitize_text(str(value)), new_x="LMARGIN", new_y="NEXT")

    def add_table(self, headers, rows, col_widths=None):
        if col_widths is None:
            col_widths = [190 / len(headers)] * len(headers)
        # Header
        self.set_font("Helvetica", "B", 9)
        self.set_fill_color(102, 126, 234)
        self.set_text_color(255, 255, 255)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 8, sanitize_text(str(h)), border=1, fill=True, align="C")
        self.ln()
        # Rows
        self.set_font("Helvetica", "", 9)
        self.set_text_color(30, 30, 30)
        for row_idx, row in enumerate(rows):
            if row_idx % 2 == 0:
                self.set_fill_color(245, 245, 250)
            else:
                self.set_fill_color(255, 255, 255)
            for i, val in enumerate(row):
                self.cell(col_widths[i], 7, sanitize_text(str(val)), border=1, fill=True, align="C")
            self.ln()


def export_data_pdf(df):
    """Generate a PDF report for the data exploration tab."""
    pdf = SentimentPDF()
    pdf.alias_nb_pages()
    pdf.add_page()

    # Summary
    pdf.section_title("Data Summary")
    pdf.key_value("Total Reviews", f"{len(df):,}")

    sentiment_counts = df["sentiment"].value_counts()
    for sent in ["Positif", "Netral", "Negatif"]:
        pdf.key_value(sent, sentiment_counts.get(sent, 0))

    if "rating" in df.columns:
        pdf.key_value("Average Rating", f"{df['rating'].mean():.2f}")

    # Sentiment distribution table
    pdf.section_title("Sentiment Distribution")
    headers = ["Sentiment", "Count", "Percentage"]
    rows = []
    total = len(df)
    for sent in ["Positif", "Netral", "Negatif"]:
        count = sentiment_counts.get(sent, 0)
        pct = f"{count / total * 100:.1f}%" if total > 0 else "0%"
        rows.append([sent, str(count), pct])
    pdf.add_table(headers, rows, col_widths=[65, 60, 65])

    # Rating distribution
    if "rating" in df.columns:
        pdf.section_title("Rating Distribution")
        rating_counts = df["rating"].value_counts().sort_index()
        headers = ["Rating", "Count", "Percentage"]
        rows = []
        for rating, count in rating_counts.items():
            pct = f"{count / total * 100:.1f}%" if total > 0 else "0%"
            rows.append([f"{rating} Star", str(count), pct])
        pdf.add_table(headers, rows, col_widths=[65, 60, 65])

    # Sample reviews
    pdf.section_title("Sample Reviews (First 20)")
    headers = ["#", "Review (truncated)", "Rating", "Sentiment"]
    rows = []
    for i, row in df.head(20).iterrows():
        text = str(row.get("review_text", ""))[:60] + "..."
        rows.append([str(len(rows) + 1), text, str(row.get("rating", "")), str(row.get("sentiment", ""))])
    pdf.add_table(headers, rows, col_widths=[12, 110, 28, 40])

    # Convert bytearray to bytes for Streamlit
    return bytes(pdf.output())


def export_model_pdf(model_result):
    """Generate a PDF report for model evaluation results."""
    pdf = SentimentPDF()
    pdf.alias_nb_pages()
    pdf.add_page()

    # Model metrics
    pdf.section_title("Model Performance Metrics")
    pdf.key_value("Accuracy", f"{model_result['accuracy'] * 100:.2f}%")
    pdf.key_value("Precision", f"{model_result['precision'] * 100:.2f}%")
    pdf.key_value("Recall", f"{model_result['recall'] * 100:.2f}%")
    pdf.key_value("F1-Score", f"{model_result['f1_score'] * 100:.2f}%")
    pdf.key_value("Train Size", str(model_result["train_size"]))
    pdf.key_value("Test Size", str(model_result["test_size"]))
    pdf.key_value("Total Features", f"{len(model_result['feature_names']):,}")
    pdf.key_value("N-Gram Range", str(model_result["ngram_range"]))

    # Classification report
    pdf.section_title("Classification Report")
    report = model_result["classification_report"]
    report_df = pd.DataFrame(report).transpose()
    labels = model_result["labels"]
    keep_rows = [r for r in report_df.index if r in labels or r == "weighted avg"]
    report_df = report_df.loc[keep_rows]

    headers = ["Class", "Precision", "Recall", "F1-Score", "Support"]
    rows = []
    for idx, row in report_df.iterrows():
        rows.append([
            str(idx),
            f"{row.get('precision', 0) * 100:.1f}%",
            f"{row.get('recall', 0) * 100:.1f}%",
            f"{row.get('f1-score', 0) * 100:.1f}%",
            str(int(row.get("support", 0))),
        ])
    pdf.add_table(headers, rows, col_widths=[42, 38, 35, 38, 37])

    # Confusion matrix
    pdf.section_title("Confusion Matrix")
    cm = model_result["confusion_matrix"]
    headers = ["Actual \\ Predicted"] + labels
    rows = []
    for i, label in enumerate(labels):
        rows.append([label] + [str(v) for v in cm[i]])
    widths = [50] + [140 / len(labels)] * len(labels)
    pdf.add_table(headers, rows, col_widths=widths)

    # Convert bytearray to bytes for Streamlit
    return bytes(pdf.output())
