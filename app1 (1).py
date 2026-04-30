import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt

st.set_page_config(page_title="Sentiment Analyzer", layout="wide")

# ── NLP PREPROCESSING ─────────────────────────────
STOPWORDS = {"a","an","the","and","or","in","on","at","to","for","of","with","is","are","was","were","be"}

def preprocess(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS]
    return " ".join(tokens)

# ── LOAD + TRAIN MODEL (CACHED) ───────────────────
@st.cache_data
def load_and_train():
    df = pd.read_csv("Womens Clothing E-Commerce Reviews.csv")

    df = df.drop(columns=["Unnamed: 0"], errors="ignore")
    df = df.dropna(subset=["Review Text"])

    df["text"] = df["Title"].fillna("") + " " + df["Review Text"]

    df["sentiment"] = df["Rating"].apply(
        lambda r: "positive" if r >= 4 else ("neutral" if r == 3 else "negative")
    )

    df["clean_text"] = df["text"].apply(preprocess)

    tfidf = TfidfVectorizer(max_features=3000)
    X = tfidf.fit_transform(df["clean_text"])

    le = LabelEncoder()
    y = le.fit_transform(df["sentiment"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=2000, class_weight="balanced")
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="weighted")

    return df, model, tfidf, le, acc, f1

df, model, tfidf, le, acc, f1 = load_and_train()

# ── UI HEADER ─────────────────────────────────────
st.title("🛍️ Women's Clothing Review Sentiment Analyzer")

st.markdown(f"""
**Model Performance**
- Accuracy: `{acc:.2f}`
- F1 Score: `{f1:.2f}`
""")

# ── SIDEBAR ──────────────────────────────────────
st.sidebar.header("Navigation")
option = st.sidebar.radio("Go to", ["Predict", "Insights"])

# ── PREDICTION PAGE ──────────────────────────────
if option == "Predict":
    st.subheader("🔍 Predict Sentiment")

    user_input = st.text_area("Enter a review")

    if st.button("Analyze"):
        clean = preprocess(user_input)
        vec = tfidf.transform([clean])
        pred = model.predict(vec)[0]
        label = le.inverse_transform([pred])[0]

        if label == "positive":
            st.success(f"Sentiment: {label}")
        elif label == "negative":
            st.error(f"Sentiment: {label}")
        else:
            st.warning(f"Sentiment: {label}")

# ── INSIGHTS PAGE ────────────────────────────────
if option == "Insights":
    st.subheader("📊 Dataset Insights")

    sentiment_counts = df["sentiment"].value_counts()

    fig, ax = plt.subplots()
    ax.bar(sentiment_counts.index, sentiment_counts.values)
    ax.set_title("Sentiment Distribution")

    st.pyplot(fig)

    st.subheader("Top Products by Reviews")

    top_products = df["Clothing ID"].value_counts().head(10)
    st.write(top_products)

    st.subheader("Average Rating by Department")

    dept = df.groupby("Department Name")["Rating"].mean().sort_values()

    fig2, ax2 = plt.subplots()
    ax2.barh(dept.index, dept.values)
    st.pyplot(fig2)

# ── FOOTER ──────────────────────────────────────
st.markdown("---")
st.caption("Built with Streamlit + ML")