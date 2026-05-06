# PUBG Mobile Sentiment Analysis Dashboard 🎮

An interactive web application built with Streamlit to analyze the sentiment of PUBG Mobile reviews from the Apple App Store. This project utilizes a **Naïve Bayes Classifier** combined with **N-Gram feature extraction** to categorize reviews into Positive, Neutral, or Negative sentiments.

## 🚀 Features

- **App Store Scraper**: Automatically fetch real-time reviews for PUBG Mobile.
- **Advanced Preprocessing**: Comprehensive text cleaning including case folding, slang normalization, stopword removal, and stemming (using PySastrawi).
- **Interactive Dashboard**:
  - **Data Exploration**: Visualize rating distributions and sentiment balance.
  - **N-Gram Comparison**: Compare performance across different N-Gram configurations (Unigram, Bigram, Trigram).
  - **Model Evaluation**: Detailed metrics including Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.
  - **Real-time Prediction**: Test the model with your own custom review text.
- **Modern UI**: Sleek dark-themed interface with Plotly visualizations.

## 🛠️ Tech Stack

- **Language**: Python 3.x
- **Web Framework**: [Streamlit](https://streamlit.io/)
- **Machine Learning**: Scikit-Learn (Multinomial Naïve Bayes)
- **NLP**: PySastrawi, NLTK, WordCloud
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Data Handling**: Pandas, NumPy

## 📦 Installation

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/yourusername/pubg-mobile-sentiment-analysis.git
   cd pubg-mobile-sentiment-analysis
   ```

2. **Create a Virtual Environment**:

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 🎮 How to Run

1. **Start the Streamlit App**:

   ```bash
   streamlit run app.py
   ```

2. **Open in Browser**:
   Navigate to `http://localhost:8501` to view the dashboard.

## 📂 Project Structure

```text
├── app.py              # Main Streamlit dashboard application
├── scraper.py          # App Store scraper and data generator
├── preprocessing.py    # Text cleaning and NLP pipeline
├── model.py            # Naïve Bayes model training and evaluation
├── requirements.txt    # Project dependencies
├── data/               # Directory for raw and processed datasets
└── models/             # Directory for saved pickle models
```

## 📊 Methodology

1. **Data Collection**: Scraping reviews via `app-store-scraper`.
2. **Labeling**: Ratings are mapped to sentiments (1-2: Negative, 3: Neutral, 4-5: Positive).
3. **Preprocessing**: Indonesian-specific NLP pipeline.
4. **Feature Extraction**: TF-IDF Vectorization with configurable N-Gram ranges.
5. **Classification**: Multinomial Naïve Bayes for efficient and accurate text classification.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

_Created with ❤️ for the PUBG Mobile Community._
