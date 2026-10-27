import streamlit as st
import os
from src.ingestion import process_pdf
from src.retriever import HybridRetriever
from src.generator import generate_answer

st.set_page_config(page_title="DocuRerank", layout="wide", page_icon="📄")

st.title("📄 DocuRerank: Enterprise Hybrid-Search RAG Engine")
st.caption("Dense Vector Search + BM25 Sparse Retrieval + Cross-Encoder Re-ranking")

with st.sidebar:
    st.header("⚙️ Configuration")
    groq_key = st.text_input("Enter Groq API Key:", type="password")
    if groq_key:
        os.environ["GROQ_API_KEY"] = groq_key

uploaded_file = st.file_uploader("Upload a PDF document to query:", type=["pdf"])

if uploaded_file:
    if "retriever" not in st.session_state or st.session_state.get("file_name") != uploaded_file.name:
        with st.spinner("Processing document and generating hybrid indices..."):
            chunks = process_pdf(uploaded_file)
            if not chunks or len(chunks) == 0:
                st.error("⚠️ No readable text found. This PDF appears to be a scanned image or empty. Please upload a PDF with selectable digital text.")
            else:
                st.session_state.retriever = HybridRetriever(chunks)
                st.session_state.file_name = uploaded_file.name
                st.success(f"Indexed {len(chunks)} chunks successfully!")

if "retriever" in st.session_state:
    query = st.text_input("Ask a question based on your document:")
    if query and st.button("Generate Answer"):
        with st.spinner("Executing hybrid retrieval, re-ranking, and grounded generation..."):
            top_contexts = st.session_state.retriever.retrieve_and_rerank(query, top_k=3)
            answer = generate_answer(query, top_contexts)

            col1, col2 = st.columns([3, 2])
            with col1:
                st.subheader("Grounded Answer")
                st.write(answer)
            with col2:
                st.subheader("Top Re-ranked Context Sources")
                for i, ctx in enumerate(top_contexts):
                    with st.expander(f"Reference Source #{i + 1}"):
                        st.write(ctx)