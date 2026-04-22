'''
UI for the retrieval application. 
Integrated with RAGPipeline for qwen2.5 generation and Tavily web tools.
'''
import streamlit as st
import sys
import csv
from pathlib import Path
from datetime import datetime, timezone

# Ensure that src and local modules are discoverable
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
from hybrid import hybrid_search 

# Import your custom pipeline classes
from rag_pipeline import RAGPipeline, HybridRetriever

st.set_page_config(
    page_title="Amazon Electronics RAG Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS STYLING
AMAZON_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: linear-gradient(180deg, #0f1111 0%, #131a22 100%); color: #f5f5f5; }

/* RAG Answer Panel */
.rag-answer-container {
    background: linear-gradient(135deg, rgba(255,153,0,0.15) 0%, rgba(19, 26, 34, 0.8) 100%);
    border: 1px solid rgba(255,153,0,0.4);
    border-radius: 20px;
    padding: 1.8rem;
    margin-bottom: 2rem;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
}
.rag-header { color: #ff9900; font-weight: 800; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 0.15em; margin-bottom: 1rem; }
.tool-badge { background: #2e7d32; color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.7rem; margin-left: 10px; vertical-align: middle; }
.rag-text { font-size: 1.1rem; line-height: 1.7; color: #ffffff; }

/* Result Cards */
.result-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-left: 4px solid #ff9900;
    border-radius: 16px;
    padding: 1.2rem;
    margin-bottom: 1rem;
}
.rank-pill { background: rgba(255,153,0,0.1); color: #ffb84d; padding: 0.3rem 0.8rem; border-radius: 999px; font-weight: 700; font-size: 0.75rem; }
.review-box { background: rgba(0,0,0,0.3); padding: 0.8rem; border-radius: 10px; font-size: 0.9rem; color: #d1d5db; margin-top: 0.8rem; }
</style>
"""
st.markdown(AMAZON_CSS, unsafe_allow_html=True)

# INITIALIZATION
@st.cache_resource
def get_pipeline():
    '''Initializes and caches the HybridRetriever-backed RAGPipeline for the Streamlit session.'''
    retriever = HybridRetriever()
    # Create the pipeline with qwen2.5 and web tools enabled
    return RAGPipeline(retriever=retriever, model="qwen2.5", use_tools=True)

rag_pipe = get_pipeline()

# UI HEADER
st.markdown('<h1 style="color:white; margin-bottom:0;">Amazon Electronics RAG</h1>', unsafe_allow_html=True)
st.markdown('<p style="color:#9aa4ad; margin-bottom:2rem;">Powered by qwen2.5 & Hybrid Search</p>', unsafe_allow_html=True)

query = st.text_input("Search or ask a question...", placeholder="e.g. Best noise cancelling headphones for under $200")
top_k = st.slider("Context Documents", 1, 10, 5)

# MODE TABS
tab_search, tab_rag = st.tabs(["🔍 Search Results", "🤖 RAG Answer"])

def render_result_card(rank, row, score, is_source=False):
    '''Renders a styled HTML product card with title, rating, and review snippet in the Streamlit UI.'''
    title = row.get("product_title", "N/A")
    rating = row.get("average_rating", "N/A")
    snippet = (row.get("review_text_200") or "No review available.")[:200]
    label = f"Source [{rank}]" if is_source else f"Rank #{rank}"
    
    st.markdown(f"""
        <div class="result-card">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.5rem;">
                <span class="rank-pill">{label}</span>
                <span style="font-size:0.8rem; color:#6b7280;">Score: {round(float(score), 4)}</span>
            </div>
            <div style="font-weight:700; font-size:1.1rem; color:white;">{title}</div>
            <div style="color:#ff9900; font-size:0.9rem; margin-top:4px;">⭐ {rating} Rating</div>
            <div class="review-box">{snippet}...</div>
        </div>
    """, unsafe_allow_html=True)

# SEARCH TAB
with tab_search:
    if query:
        with st.spinner("Retrieving products..."):
            # Using the retriever instance inside the pipeline for consistency
            results = rag_pipe.retriever.invoke(query, top_k=top_k)
            if results:
                for i, (row, score) in enumerate(results, 1):
                    render_result_card(i, row, score)
            else:
                st.warning("No products matched your query.")

# RAG TAB
with tab_rag:
    if query:
        with st.spinner("Phi4-mini is thinking..."):
            # Invoke the full RAG pipeline
            response = rag_pipe.invoke(query, top_k=top_k, system_prompt_version='V3')
            
            # Display Answer Panel
            tool_info = f'<span class="tool-badge">Used {response["tool_used"]}</span>' if response["tool_used"] else ""
            st.markdown(f"""
                <div class="rag-answer-container">
                    <div class="rag-header">AI Response {tool_info}</div>
                    <div class="rag-text">{response['llm_answer']}</div>
                </div>
            """, unsafe_allow_html=True)
            
            # Display Attributed Sources
            st.markdown("### Cited Sources")
            for i, (row, score) in enumerate(response['retrieved_docs'], 1):
                render_result_card(i, row, score, is_source=True)
    else:
        st.info("Please enter a query to generate an AI answer.")
