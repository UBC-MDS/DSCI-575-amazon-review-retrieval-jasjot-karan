'''
Implements a RAG Pipeline using a Custom Retriever with Semantic Seach using FAISS HNSW indexes from semantic.py, 
'''
from semantic import (
    load_sentence_transformer_smodel,
    load_or_build_semantic_artifacts,
    CORPUS_PATH,
    FAISS_INDEX_PATH,
    METADATA_PATH,
    CHUNK_SIZE,
    EMBED_BATCH_SIZE,
    semantic_search
)

from prompts import (
    SYSTEM_PROMPT_V1,
    SYSTEM_PROMPT_V2,
    SYSTEM_PROMPT_V3,
    build_prompt
)

def build_context(docs: list[tuple[dict, float]]): 
    '''
    Turns the retrieved content from a list of tuples: (product_metadata dict, score) into a single string that contains all top k ranked products' metadata
    '''
    all_context = []

    for rank, (doc, _) in enumerate(docs, start = 1):
        context = (
            f"[Product rank: {rank}]\n"
            f"ASIN: {doc.get('parent_asin', 'N/A')}\n"
            f"Product Title: {doc.get('product_title', 'N/A')}\n"
            f"Category: {doc.get('main_category', 'N/A')}\n"
            f"Store: {doc.get('store', 'N/A')}\n"
            f"Price: {doc.get('price', 'N/A')}\n"
            f"Average Rating: {doc.get('average_rating', 'N/A')}\n"
            f"Description: {(doc.get('description') or '')[:500]}\n"
            f"Review snippets: {(doc.get('review_text_200') or '').strip()[:500]}\n"
        )
    
        all_context.append(context)

    return "\n\n=========\n\n".join(all_context)
class SemanticRetriever:
    def __init__(self, max_rows = None):
        # initialize the retriever when it is called by loading or building our semantic artifacts needed for semantic retrieval (FAISS index and metadata rows for output)
        # this avoids re-loading our FAISS index and metadata rows, along with our sentece transformer "all-MiniLM-L6-v2" on each call to the Retiever so the FAISS Index, metadata rows, and sentence transformer are loaded for subsequent calls
        self.faiss_index, self.metadata_rows =  load_or_build_semantic_artifacts(
            corpus_path = CORPUS_PATH,
            index_path = FAISS_INDEX_PATH,
            metadata_path = METADATA_PATH,
            chunk_size = CHUNK_SIZE,
            batch_size = EMBED_BATCH_SIZE
        )

        self.embedding_model = load_sentence_transformer_smodel() # uses our default  "all-MiniLM-L6-v2" semtemce tramsformer model

    def invoke(self, query: str, top_k: int = 5):
        '''
        Calls the semantic_search() function from semantic.py and returns a list of sorted tuples by similarity score in desc order, where each tuple at index i contains the i'th scoring product, along with that products similarity score.
        '''
        return semantic_search(
            query = query,
            index = self.faiss_index,
            metadata_rows = self.metadata_rows,
            model = self.embedding_model,
            top_k = top_k
        )

if __name__ == "__main__":

    retriever = SemanticRetriever()

    docs = retriever.invoke(
        'Mechanical Keyboard that is good for coding and makes a nice, clickly sound when typing',
        top_k = 5
    )

    product_context = build_context(docs)
    print(product_context)

c;ass 