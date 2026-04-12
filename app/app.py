import streamlit as st
import sys
import os
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from bm25 import (
    bm25_search, 
    load_or_build_search_artifacts,
    CORPUS_PATH as BM25_CORPUS_PATH,
    TOKENIZED_PATH,
    METADATA_PATH as BM25_META_PATH,
    BM25_PATH,
    CHUNK_SIZE as BM25_CHUNK_SIZE
)

from semantic import (
    semantic_search, 
    load_or_build_semantic_artifacts,
    load_sentence_transformer_smodel,
    CORPUS_PATH as SEMANTIC_CORPUS_PATH,
    FAISS_INDEX_PATH,
    METADATA_PATH as SEMANTIC_META_PATH,
    CHUNK_SIZE as SEMANTIC_CHUNK_SIZE,
    EMBED_BATCH_SIZE
)

st.set_page_config(page_title="Amazon Electronics Product Search", layout="wide")

# DATA LOADING
@st.cache_resource
def initialize_models():
    # Load BM25 artifacts
    bm25_index, bm_25_metadata_rows = load_or_build_search_artifacts(
        corpus_path = BM25_CORPUS_PATH,
        tokenized_path = TOKENIZED_PATH, 
        metadata_path = BM25_META_PATH,
        bm25_path = BM25_PATH,
        chunk_size = BM25_CHUNK_SIZE,
        max_rows = None
    )
    
    # Load semantic artifacts
    faiss_index, semantic_metadata_rows = load_or_build_semantic_artifacts(
        corpus_path = SEMANTIC_CORPUS_PATH,
        index_path = FAISS_INDEX_PATH,
        metadata_path = SEMANTIC_META_PATH,
        chunk_size = SEMANTIC_CHUNK_SIZE,
        batch_size = EMBED_BATCH_SIZE,
        max_rows = None
    )

    model = load_sentence_transformer_smodel()
    
    return bm25_index, bm_25_metadata_rows, faiss_index, semantic_metadata_rows, model

bm25_index, bm_25_metadata_rows, faiss_index, semantic_metadata_rows, model = initialize_models()

# UI LAYOUT
st.title("🛍️ Amazon Electronics Product Retrieval")
st.markdown("Compare keyword-based **BM25** and **Semantic Search** results side-by-side.")

query = st.text_input(
    "Enter your product search query:", 
    placeholder = "e.g., lightweight laptop for travel"
)

search_mode = st.radio(
    "Search mode",
    options = ["BM25", "Semantic Search"],
    horizontal = True
)

top_k = st.sidebar.slider("Number of results to show", 1, 10, 3) # make 3 the default

if query:
    if search_mode == "BM25":
        st.header("BM25 Results")
        
        bm25_results = bm25_search(
            query = query, 
            bm25 = bm25_index, 
            metadata_rows = bm_25_metadata_rows, 
            top_k = top_k
        )
        
        for rank, (row, score) in enumerate(bm25_results, start = 1):
            st.subheader(f"Rank {rank}")
            # Displaying the 4 specific things from the screenshot
            st.write(f"**Score:** {round(float(score), 4)}")
            st.write(f"**Title:** {row.get('product_title', 'N/A')}")
            st.write(f"**Rating:** {row.get('average_rating', 'N/A')}")
            st.write(f"**Main Category:** {row.get('main_category', 'N/A')}")
            st.write(f"**Review Snippet:** {(row.get('review_text_200') or '')[:200]}")
            st.divider()

    elif search_mode == "Semantic Search":
        st.header("Semantic Search Results")

        sem_results = semantic_search(
            query = query, 
            index = faiss_index, 
            metadata_rows = semantic_metadata_rows, 
            model = model, 
            top_k = top_k
        )
        
        for rank, (row, score) in enumerate(sem_results, start=1):
            st.subheader(f"Rank {rank}")
            st.write(f"**Similarity Score:** {round(float(score), 4)}")
            st.write(f"**Title:** {row.get('product_title', 'N/A')}")
            st.write(f"**Rating:** {row.get('average_rating', 'N/A')}")
            st.write(f"**Main Category:** {row.get('main_category', 'N/A')}")
            st.write(f"**Review Snippet:** {(row.get('review_text_200') or '')[:200]}")
            st.divider()