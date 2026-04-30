# 🛍️ Women's Clothing Review Sentiment Analyzer

A machine learning project that classifies customer reviews from a Women's E-Commerce Clothing dataset into **positive**, **neutral**, or **negative** sentiments. Includes a full NLP pipeline, multi-model benchmarking, and an interactive Streamlit web app.

---

## 📌 Project Overview

This project builds a text classification system on top of 22,641 real clothing reviews spanning 1,179 unique products. It explores the complete ML lifecycle — from raw text preprocessing to model selection and deployment via a live web interface.

---

## ✨ Features

- **Custom NLP Pipeline** — lowercasing, contraction expansion, stopword removal (with negation preservation), and rule-based lemmatization, all without external NLP libraries
- **TF-IDF Vectorization** — unigrams + bigrams, sublinear TF scaling, top 3,000 features
- **Multi-Model Benchmarking** — Logistic Regression, Naive Bayes, and Linear SVC trained and evaluated side-by-side
- **Automatic Best-Model Selection** — picks the best model by weighted F1 score
- **Keyword Extraction** — identifies top sentiment-driving words per class using model coefficients
- **Top Product Insights** — per-product average rating and sentiment breakdown
- **Interactive Streamlit App** — real-time sentiment prediction + dataset visualizations

---

## 📊 Model Results

| Model               | Accuracy | Weighted F1 |
|---------------------|----------|-------------|
| Logistic Regression | 0.7903   | 0.8090      |
| Naive Bayes         | 0.8066   | 0.7608      |
| **Linear SVC**      | **0.8110** | **0.8152** |

> **Best Model: Linear SVC** (selected automatically based on weighted F1)

---

## 🗂️ Project Structure

```
├── Sentiment_Analyzer.ipynb        # Core ML pipeline & analysis
├── app1.py                         # Streamlit web application
├── Womens Clothing E-Commerce Reviews.csv   # Dataset (not included)
├── results.json                    # Saved best model + keywords (auto-generated)
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/your-username/sentiment-analyzer.git
cd sentiment-analyzer
```

### 2. Install dependencies
```bash
pip install streamlit pandas numpy scikit-learn matplotlib
```

### 3. Add the dataset
Download the [Women's E-Commerce Clothing Reviews](https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews) dataset from Kaggle and place it in the project root as:
```
Womens Clothing E-Commerce Reviews.csv
```

### 4. Run the notebook
```bash
jupyter notebook Sentiment_Analyzer.ipynb
```

### 5. Launch the Streamlit app
```bash
streamlit run app1.py
```

---

## 🖥️ App Demo

The Streamlit app has two pages:

**Predict** — Enter any clothing review and get an instant sentiment prediction.

**Insights** — Explore the dataset through:
- Sentiment distribution bar chart
- Top 10 most-reviewed products
- Average rating by department (horizontal bar chart)

---

## 🧠 NLP Pipeline Details

| Step | Description |
|------|-------------|
| Lowercasing | Normalize all text to lowercase |
| Contraction Expansion | `won't` → `will not`, `can't` → `cannot`, etc. |
| Special Char Removal | Strip punctuation and digits |
| Tokenization | Whitespace splitting |
| Stopword Removal | Custom stopword list; negations (`not`, `never`, `no`) are **preserved** |
| Lemmatization | Regex-based suffix stripping (no external libraries) |

---

## 📦 Dependencies

- Python 3.8+
- `pandas`, `numpy`
- `scikit-learn`
- `streamlit`
- `matplotlib`

---

## 📁 Dataset

**Women's E-Commerce Clothing Reviews** — publicly available on [Kaggle](https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews).

- 23,486 rows, 10 feature columns
- Key fields used: `Review Text`, `Title`, `Rating`, `Clothing ID`, `Department Name`, `Division Name`, `Class Name`

---

## 🔮 Future Improvements

- [ ] Add confidence scores to predictions
- [ ] Swap TF-IDF for sentence embeddings (e.g. `sentence-transformers`)
- [ ] Support CSV batch upload for bulk prediction
- [ ] Deploy to Streamlit Cloud or Hugging Face Spaces

---

## 👤 Author

Built as part of an ML/NLP learning project.  
Feel free to fork, star ⭐, and contribute!
