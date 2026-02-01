import streamlit as st
from pypdf import PdfReader
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

# ---------- PDF TEXT EXTRACTION ----------
def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

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
st.set_page_config(page_title="PDF Text Summarizer", layout="centered")
st.title("📄 PDF Text Summarizer (Machine Learning)")

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
num_sentences = st.slider("Number of summary sentences", 3, 15, 5)

if uploaded_file:
    with st.spinner("Reading PDF..."):
        text = extract_text_from_pdf(uploaded_file)

    if not text.strip():
        st.error("No readable text found in PDF.")
    else:
        with st.spinner("Generating summary..."):
            summary = summarize_text(text, num_sentences)

        st.subheader("✅ Summary")
        st.write(summary)

        with st.expander("📜 Full Extracted Text"):
            st.write(text)
