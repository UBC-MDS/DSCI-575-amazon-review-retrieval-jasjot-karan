'''
UI for the retrieval application. 
Attribution: Used ChatGPT 5 to generate HTML code that matches the Amazon theme, along with actual cards to make the UI design cleaner.
'''
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

st.set_page_config(
    page_title="Amazon Electronics Product Retrieval",
    layout="wide",
    initial_sidebar_state="collapsed"
)

AMAZON_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(180deg, #0f1111 0%, #131a22 100%);
    color: #f5f5f5;
}

/* main container */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}

/* hide some default streamlit chrome */
#MainMenu, footer, header {
    visibility: hidden;
}

/* hide weird default status/spinner icons where possible */
[data-testid="stDecoration"] {
    display: none;
}

/* title hero */
.hero-wrap {
    background: linear-gradient(135deg, rgba(255,153,0,0.16), rgba(255,153,0,0.04));
    border: 1px solid rgba(255,153,0,0.18);
    border-radius: 24px;
    padding: 2rem 2rem 1.5rem 2rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 10px 30px rgba(0,0,0,0.30);
}

.hero-badge {
    display: inline-block;
    background: rgba(255,153,0,0.15);
    color: #ffb84d;
    border: 1px solid rgba(255,153,0,0.25);
    padding: 0.35rem 0.75rem;
    border-radius: 999px;
    font-size: 0.82rem;
    font-weight: 600;
    margin-bottom: 1rem;
}

.hero-title {
    font-size: 3rem;
    font-weight: 800;
    line-height: 1.05;
    color: #ffffff;
    margin: 0 0 0.75rem 0;
    letter-spacing: -0.02em;
}

.hero-subtitle {
    font-size: 1.05rem;
    color: #d5d9d9;
    margin: 0;
    max-width: 850px;
}

/* labels */
.stTextInput label, .stRadio label, .stSlider label {
    color: #f5f5f5 !important;
    font-weight: 600 !important;
}

/* text input */
.stTextInput input {
    background-color: #1a1a1a !important;
    color: #ffffff !important;
    border: 1px solid #2f3b46 !important;
    border-radius: 14px !important;
    padding: 0.9rem 1rem !important;
    font-size: 1rem !important;
}

.stTextInput input::placeholder {
    color: #9aa4ad !important;
}

/* radio buttons container */
div[role="radiogroup"] {
    background: rgba(255,255,255,0.03);
    padding: 0.85rem 1rem;
    border-radius: 16px;
    border: 1px solid rgba(255,255,255,0.08);
    gap: 1rem;
}

/* slider */
[data-testid="stSidebar"] {
    background: #111827;
}

/* track (background line) */
.stSlider [data-baseweb="slider"] > div > div {
    background: #2f3b46 !important;
}

/* filled track (left side of thumb) */
.stSlider [data-baseweb="slider"] > div > div > div {
    background: #ff9900 !important;  /* Amazon orange */
}

/* slider thumb (circle) */
.stSlider [data-baseweb="slider"] div[role="slider"] {
    background-color: #ff9900 !important;
    border: 2px solid #ffb84d !important;
    box-shadow: 0 0 0 4px rgba(255,153,0,0.15) !important;
}

/* hover effect */
.stSlider [data-baseweb="slider"] div[role="slider"]:hover {
    box-shadow: 0 0 0 6px rgba(255,153,0,0.25) !important;
}

/* number text (1-10) */
.stSlider span {
    color: #d5d9d9 !important;
}

/* custom card */
.result-card {
    background: linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.02));
    border: 1px solid rgba(255,255,255,0.08);
    border-left: 4px solid #ff9900;
    border-radius: 20px;
    padding: 1.2rem 1.2rem 1rem 1.2rem;
    margin-bottom: 1rem;
    box-shadow: 0 10px 25px rgba(0,0,0,0.22);
}

.result-top {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 1rem;
    margin-bottom: 0.75rem;
    flex-wrap: wrap;
}

.rank-pill {
    display: inline-block;
    background: rgba(255,153,0,0.14);
    color: #ffb84d;
    border: 1px solid rgba(255,153,0,0.26);
    padding: 0.28rem 0.7rem;
    border-radius: 999px;
    font-size: 0.82rem;
    font-weight: 700;
}

.score-pill {
    display: inline-block;
    background: rgba(255,255,255,0.06);
    color: #f3f4f6;
    border: 1px solid rgba(255,255,255,0.08);
    padding: 0.28rem 0.7rem;
    border-radius: 999px;
    font-size: 0.82rem;
    font-weight: 600;
}

.result-title {
    font-size: 1.18rem;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 0.65rem;
    line-height: 1.4;
}

.meta-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.6rem;
    margin-bottom: 0.8rem;
}

.meta-chip {
    background: rgba(255,255,255,0.05);
    color: #d5d9d9;
    border: 1px solid rgba(255,255,255,0.06);
    padding: 0.35rem 0.7rem;
    border-radius: 999px;
    font-size: 0.84rem;
}

.review-box {
    background: rgba(0,0,0,0.22);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 14px;
    padding: 0.9rem 1rem;
    color: #e5e7eb;
    line-height: 1.55;
    font-size: 0.96rem;
}

/* headers */
.section-title {
    font-size: 1.4rem;
    font-weight: 750;
    color: #ffffff;
    margin: 1rem 0 1rem 0;
}

/* remove divider heaviness */
hr {
    border-color: rgba(255,255,255,0.08);
}
</style>
"""
st.markdown(AMAZON_CSS, unsafe_allow_html=True)

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
st.markdown(
    """
    <div class="hero-wrap">
        <div class="hero-badge">Amazon Electronics Product Retrieval</div>
        <div class="hero-title">Amazon Electronics Product Retrieval</div>
        <p class="hero-subtitle">
            Search electronics products using either keyword-based BM25 or semantic retrieval.
            Clean, fast, and built to feel more production-ready.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

col1, col2 = st.columns([4, 1])

with col1:
    query = st.text_input(
        "Enter your product search query",
        placeholder="e.g. lightweight laptop for travel, noise cancelling headphones, gaming monitor"
    )

with col2:
    top_k = st.slider("Results", 1, 10, 3)

search_mode = st.radio(
    "Search mode",
    options=["BM25", "Semantic Search"],
    horizontal=True
)

def render_result_card(rank, row, score, score_label="Score"):
    title = row.get("product_title", "N/A")
    rating = row.get("average_rating", "N/A")
    category = row.get("main_category", "N/A")
    snippet = (row.get("review_text_200") or "No review snippet available.")[:200]

    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-top">
                <span class="rank-pill">Rank #{rank}</span>
                <span class="score-pill">{score_label}: {round(float(score), 4)}</span>
            </div>
            <div class="result-title">{title}</div>
            <div class="meta-row">
                <span class="meta-chip">⭐ Rating: {rating}</span>
                <span class="meta-chip">📦 Category: {category}</span>
            </div>
            <div class="review-box">
                {snippet}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

if query:
    if search_mode == "BM25":
        st.markdown('<div class="section-title">BM25 Results</div>', unsafe_allow_html=True)

        with st.spinner("Searching BM25 index..."):
            bm25_results = bm25_search(
                query=query,
                bm25=bm25_index,
                metadata_rows=bm_25_metadata_rows,
                top_k=top_k
            )

        if bm25_results:
            for rank, (row, score) in enumerate(bm25_results, start=1):
                render_result_card(rank, row, score, score_label="BM25 Score")
        else:
            st.warning("No BM25 results found.")

    elif search_mode == "Semantic Search":
        st.markdown('<div class="section-title">Semantic Search Results</div>', unsafe_allow_html=True)

        with st.spinner("Running semantic retrieval..."):
            sem_results = semantic_search(
                query=query,
                index=faiss_index,
                metadata_rows=semantic_metadata_rows,
                model=model,
                top_k=top_k
            )

        if sem_results:
            for rank, (row, score) in enumerate(sem_results, start=1):
                render_result_card(rank, row, score, score_label="Similarity")
        else:
            st.warning("No semantic search results found.")
else:
    st.markdown(
        """
        <div class="result-card" style="border-left: 4px solid #232f3e;">
            <div class="result-title">Try a search</div>
            <div class="review-box">
                Example queries:
                <br><br>
                • wireless earbuds for gym<br>
                • travel laptop with long battery life<br>
                • gaming mouse for fps games
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )