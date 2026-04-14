'''
Implements a RAG Pipeline using a Custom Retriever with Semantic Seach using FAISS HNSW indexes from semantic.py, 
'''
from ollama import chat
from dotenv import load_dotenv 

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

class RAGPipeline: 
    def __init__(self, retriever, model = "phi4-mini"):
        self.retriever =  retriever
        self.model = model

    def invoke(self, query: str, top_k: int, system_prompt_version: str = 'V3', max_rows: int = None):
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
        
        # return the user query, answer, retrieved top k documents for the user query 
        return {
            "query": query,
            "llm_answer": response.message.content,
            "retrieved_docs": docs,
            "prompt_version": system_prompt_version
        }

if __name__ == "__main__":
    retriever = SemanticRetriever()
    rag_pipeline = RAGPipeline(retriever = retriever, model = "phi4-mini")

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
