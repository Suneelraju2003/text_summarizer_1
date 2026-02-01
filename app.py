import streamlit as st
import nltk
import numpy as np
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------- NLTK SETUP ----------
nltk.download("punkt")
nltk.download("stopwords")
stop_words = stopwords.words("english")

# ---------- ML TEXT SUMMARIZER ----------
def summarize_text(text, num_sentences=5):
    sentences = sent_tokenize(text)

    if len(sentences) <= num_sentences:
        return text

    vectorizer = TfidfVectorizer(stop_words=stop_words)
    tfidf = vectorizer.fit_transform(sentences)

    similarity_matrix = cosine_similarity(tfidf)
    scores = similarity_matrix.sum(axis=1)

    ranked_sentences = np.argsort(scores)[::-1]
    selected = sorted(ranked_sentences[:num_sentences])

    summary = " ".join([sentences[i] for i in selected])
    return summary

# ---------- STREAMLIT UI ----------
st.set_page_config(page_title="Text Summarizer", layout="centered")
st.title("📝 Text Summarizer (Machine Learning)")
st.write("Paste any long text and get a short summary")

text_input = st.text_area(
    "Enter text to summarize:",
    height=300,
    placeholder="Paste article, notes, research paper text here..."
)

num_sentences = st.slider(
    "Summary length (number of sentences)",
    2, 15, 5
)

if st.button("Summarize"):
    if not text_input.strip():
        st.warning("Please enter some text.")
    else:
        with st.spinner("Generating summary..."):
            summary = summarize_text(text_input, num_sentences)

        st.subheader("✅ Summary")
        st.write(summary)

        with st.expander("📜 Original Text"):
            st.write(text_input)
