'''
Implements a RAG Pipeline using a Custom Retriever with Semantic Seach using FAISS HNSW indexes from semantic.py, 
'''
from ollama import chat
from dotenv import load_dotenv 

from src.bm25 import (
    load_or_build_search_artifacts, 
    bm25_search,
    TOKENIZED_PATH,
    BM25_PATH as BM25_INDEX_PATH,
    CHUNK_SIZE as BM25_CHUNK_SIZE
)

from src.semantic import (
    load_sentence_transformer_smodel,
    load_or_build_semantic_artifacts,
    CORPUS_PATH,
    FAISS_INDEX_PATH,
    METADATA_PATH,
    CHUNK_SIZE,
    EMBED_BATCH_SIZE,
    semantic_search
)

from src.hybrid import hybrid_search

from src.prompts import (
    SYSTEM_PROMPT_V1,
    SYSTEM_PROMPT_V2,
    SYSTEM_PROMPT_V3,
    build_prompt
)

load_dotenv()

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
    '''
    Class defines a semantic retriever that uses cosine similarity search using FAISS HNSW indexes. 

    Class attributes: 
        - self.faiss_index: FAISS index already built/processed from load_or_build_semantic_artifacts(...) in semantic.py
        - self.metadata_rows: product metadata rows, where each row is a dictionary containing (product attribute: value). Row index positions of metadata_rows from load_or_build_semantic_artifacts(...) match the index positions of the embeddings in the index, allowing us to find the product metadata row that corresponds to the embedding at index i.
        - self.embedding_model: set to 'all-MiniLM-L6-v2 model' embedding model from load_sentence_transformer_smodel() in semantic.py
    '''
    def __init__(self):
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
    
class BM25Retriever:
    '''
    Class defines a BM25 retriever that uses the BM25 metric to retrieve relevant product metadata matching a user query. 

    Class attributes: 
        - self.bm25_index: BM25 index already built/processed from load_or_build_search_artifacts(...) in bm25.py
        - self.metadata_rows: product metadata rows, where each row is a dictionary containing (product attribute: value). Index positions of the top k scores in bm25_index will be mapped to the top k index positions in metadata_rows to get the product metadata for top k retrieved results.
    '''
    def __init__(self):
        # initialize the retriever when it is called by loading or building our semantic artifacts needed for BM25 retrieval so they do not have to be re-called constantly after a BM25 Retriever instance is initialized.
        self.bm25_index, self.bm25_metadata_rows = load_or_build_search_artifacts(
            corpus_path = CORPUS_PATH,
            tokenized_path = TOKENIZED_PATH,
            metadata_path = METADATA_PATH,
            bm25_path = BM25_INDEX_PATH,
            chunk_size = BM25_CHUNK_SIZE
        )

    def invoke(self, query: str, top_k: int = 5):
        '''
        Calls the bm25_search() function from bm25.py and returns a list of sorted tuples by similarity score in desc order, where each tuple at index i contains the i'th scoring product metadata, along with that products similarity score.
        '''
        return bm25_search(
            query = query,
            bm25 = self.bm25_index,
            metadata_rows = self.bm25_metadata_rows,
            top_k = top_k
        )

class HybridRetriever:
    '''
    Hybrid Retriever that combines BM25 keyword search and semantic vector search using Reciprocal
    Rank Fusion (RRF). Uses top_k() from src/hybrid.py. 

    Instantiated with: BM25Retriever and SemanticRetriever objects that each include (bm25_index and bm25_metadata_rows attributes) and (faiss_index, semantic_metadata_rows, and model) attrbibutes to pass precomputed objects into the hybrid_search() functions 
    '''
    def __init__(self):
        self.bm25_retriever = BM25Retriever()
        self.semantic_retriever = SemanticRetriever()

    def invoke(self, query: str, top_k: int = 5, candidate_multiplier: int = 3, rrf_k: int = 60):
        '''
        invoke method calls the hybrid_search(...) function with the passed in (bm25_index and bm25_metadata_rows) attributes from BM25Retriever, (faiss_index, semantic_metadata_rows, and model) attributes from SemanticRetriever, along with the query, top_k, candidate_multiplier and rrf_k hyperparameters
        '''
        return hybrid_search(
                query = query, 
                bm25_index = self.bm25_retriever.bm25_index,
                bm25_metadata_rows = self.bm25_retriever.bm25_metadata_rows,
                faiss_index = self.semantic_retriever.faiss_index,
                semantic_metadata_rows = self.semantic_retriever.metadata_rows,
                model = self.semantic_retriever.embedding_model,
                top_k = top_k,
                candidate_multiplier = candidate_multiplier, # retrieves more than top k from each method before fusing the scores so the hybrid score has a better signal ex) If one method ranks a doc 2 and the otehr ranks it 12, having no candidate_multiplier results in only top 5 rankings compared compared, which could miss some similarities
                rrf_k = rrf_k # RRF smoothing constant
        )
class RAGPipeline: 
    def __init__(self, retriever, model = "phi4-mini"):
        self.retriever =  retriever
        self.model = model

    def invoke(self, query: str, top_k: int, system_prompt_version: str = 'V3'):
        docs = self.retriever.invoke(query = query, top_k = top_k)
        context = build_context(docs)
        system_prompt, user_message = build_prompt(query = query, context = context, prompt_version = system_prompt_version)

        # call the phi4-mini model using ollama
        # Referenced Ollama docs: https://ollama.com/library/phi4-mini
        response = chat(
            model = self.model,
            messages = [
                {'role': 'system', 'content': system_prompt}, # system prompt sent to the LLM
                {'role': 'user', 'content': user_message} # user message including the context to answer the user query with 
            ],
            options = {
                "temperature": 0.0 # set temperature to 0 so we get more deterministic results and the most likely next tokens for each output without any randomness since we want grounded responses
            }
        )
        
        # return the user query, answer, retrieved top k documents for the user query (to possibly extend this and use it for quantatative evaluation such as faithfulness -> see if llm_answer contains actual product context from retrieved_docs)
        return {
            "query": query,
            "llm_answer": response.message.content,
            "retrieved_docs": docs,
            "prompt_version": system_prompt_version
        }
 
if __name__ == "__main__":
    semantic_retriever = SemanticRetriever()
    hybrid_retriever = HybridRetriever()
    rag_pipeline = RAGPipeline(retriever = hybrid_retriever, model = "phi4-mini")

    # test query for our experiment
    query = 'Mechanical Keyboard that is good for coding'

    result = rag_pipeline.invoke(
        query = query,
        top_k = 5,
        system_prompt_version = 'V3'
    )

    print(f"\n======= QUERY =======\n{result['query']}\n")

    print(f"\n======= LLM ANSWER =======\n{result['llm_answer']}\n")

    print("======= TOP 5 RETRIEVED PRODUCTS =======")
    for rank, (doc, score) in enumerate(result['retrieved_docs'], start = 1):
        print(f"\n[Rank {rank}] Score: {score:.4f}")
        print(f"  ASIN:   {doc.get('parent_asin')}")
        print(f"  Title:  {doc.get('product_title')}")
        print(f"  Rating: {doc.get('average_rating')}")
        print(f"  Price: {doc.get('price')}")
        print(f"  Description: {doc.get('description')}")
        print(f"  Review snippets: {doc.get("review_text_200") or ""}")
