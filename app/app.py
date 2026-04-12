import streamlit as st
import sys
import os
from pathlib import Path

# Add the 'src' directory to the system path so we can import your modules
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from bm25 import bm25_search, load_or_build_search_artifacts
from semantic import semantic_search, load_faiss_index_and_metadata, load_sentence_transformer_smodel

st.set_page_config(page_title="Amazon Retrieval Dashboard", layout="wide")

# DATA LOADING
@st.cache_resource
def initialize_models():
    # Load BM25 artifacts
    # These functions use Path(__file__) internally to find your 'data/processed' folder
    _, bm25, metadata_rows_bm25 = load_or_build_search_artifacts()
    
    # Load Semantic artifacts
    index, metadata_rows_sem = load_faiss_index_and_metadata()
    model = load_sentence_transformer_smodel()
    
    return bm25, metadata_rows_bm25, index, metadata_rows_sem, model

bm25, metadata_bm25, index, metadata_sem, model = initialize_models()

# UI LAYOUT
st.title("🛍️ Amazon Product Retrieval")
st.markdown("Compare keyword-based **BM25** and **Semantic Search** results side-by-side.")

query = st.text_input("Enter your product search query:", placeholder="e.g., lightweight laptop for travel")
top_k = st.sidebar.slider("Number of results to show", 1, 10, 5)

if query:
    col1, col2 = st.columns(2)

    with col1:
        st.header("🔍 BM25 Results")
        # Use the search function from your bm25.py script
        bm25_results = bm25_search(query, bm25, metadata_bm25, top_k=top_k)
        
        for rank, (row, score) in enumerate(bm25_results, start=1):
            st.subheader(f"Rank {rank}")
            st.write(f"**Score:** {round(float(score), 4)}")
            st.write(f"**Title:** {row.get('product_title', 'N/A')}")
            st.write(f"**Description:** {row.get('description', 'N/A')[:250]}...")
            # Using the 200-char review snippet you processed
            st.write(f"**Review:** {(row.get('review_text_200') or '')[:200]}")
            st.divider()

    with col2:
        st.header("🧠 Semantic Search Results")
        # Use the search function from your semantic.py script
        sem_results = semantic_search(query, index, metadata_sem, model, top_k=top_k)
        
        for rank, (row, score) in enumerate(sem_results, start=1):
            st.subheader(f"Rank {rank}")
            st.write(f"**Distance:** {round(float(score), 4)}")
            st.write(f"**Title:** {row.get('product_title', 'N/A')}")
            st.write(f"**Description:** {row.get('description', 'N/A')[:250]}...")
            st.write(f"**Review:** {(row.get('review_text_200') or '')[:200]}")
            st.divider()
