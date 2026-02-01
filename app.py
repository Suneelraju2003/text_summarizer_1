
import streamlit as st
import PyPDF2
from transformers import pipeline

# Load a pre-trained summarization model globally to avoid re-loading on each rerun
@st.cache_resource
def get_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

summarizer = get_summarizer()

def extract_text_from_pdf(pdf_file):
    text = ""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text() or ""
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
    return text

def summarize_text(text, min_length_percentage=20, max_length_percentage=80):
    if not text.strip():
        return ""

    words = text.split()
    num_words = len(words)
    min_length = max(10, int(num_words * (min_length_percentage / 100)))
    max_length = max(min_length + 10, int(num_words * (max_length_percentage / 100)))

    max_length = min(max_length, 1024) 
    min_length = min(min_length, max_length - 5) if max_length > 5 else 0

    if num_words < 50:
        min_length = min(10, num_words // 2)
        max_length = min(max(min_length + 5, num_words -1), 50)

    try:
        summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text']
        return summary
    except Exception as e:
        return f"Error during summarization: {e}"


def main():
    st.set_page_config(page_title="PDF & Text Summarizer", layout="wide")
    st.title("PDF & Text Summarizer")

    st.markdown("---<br>", unsafe_allow_html=True)
    st.subheader("Upload a PDF or Paste Text for Summarization")

    pdf_file = st.file_uploader("Upload PDF", type=["pdf"])
    text_input = st.text_area("Or paste your text here:", height=200)

    st.markdown("---<br>", unsafe_allow_html=True)
    st.subheader("Summary Length Control")
    col1, col2 = st.columns(2)
    with col1:
        min_len_perc = st.slider("Minimum Summary Length (% of original)", 5, 95, 20)
    with col2:
        max_len_perc = st.slider("Maximum Summary Length (% of original)", 5, 95, 80)

    summarize_button = st.button("Summarize")

    st.markdown("---<br>", unsafe_allow_html=True)
    st.subheader("Generated Summary")

    if summarize_button:
        input_text = ""
        if pdf_file is not None:
            st.info("Extracting text from PDF...")
            input_text = extract_text_from_pdf(pdf_file)
            if not input_text:
                st.warning("Could not extract text from PDF. Please ensure it's not an image-based PDF or try pasting text.")
        elif text_input:
            input_text = text_input
        
        if input_text:
            with st.spinner("Generating summary..."):
                summary = summarize_text(input_text, min_length_percentage=min_len_perc, max_length_percentage=max_len_perc)
                st.success("Summary Generated!")
                st.write(summary)
        else:
            st.warning("Please upload a PDF or paste some text to summarize.")

if __name__ == "__main__":
    main()
