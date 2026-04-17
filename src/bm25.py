'''
Implements a BM25 keyoword based retreival system using LangChain
Following docs were used for help: https://docs.langchain.com/oss/python/integrations/retrievers/bm25
'''
import gc
import numpy as np
from rank_bm25 import BM25Okapi
from pathlib import Path
from src.utils import tokenize, get_total_rows, load_pickle_if_valid, save_pickle, load_tokenized_corpus_and_metadata_in_chunks, META_COLS

# make sure the data can be processed regardless of the directory we are in
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data/processed"

TOKENIZED_PATH = DATA_DIR / "tokenized_corpus.pkl"
BM25_PATH = DATA_DIR / "bm25_index.pkl"
CORPUS_PATH = DATA_DIR / "retrieval_corpus.parquet"
METADATA_PATH = DATA_DIR / "metadata_rows.pkl"

CHUNK_SIZE = 10000

def load_or_build_corpus_artifacts(
    corpus_path: str,
    tokenized_path: str = TOKENIZED_PATH,
    metadata_path: str = METADATA_PATH,
    chunk_size: int = CHUNK_SIZE,
    max_rows: int | None = None
) -> tuple[list[list[str]], list[dict]]:
    '''
    Load tokenized corpus and metadata rows if they already exist. 
    Else, load the raw texts and metadata rows, tokenize texts for BM25 and save both
    '''
    tokenized_corpus = load_pickle_if_valid(tokenized_path)
    metadata_rows = load_pickle_if_valid(metadata_path)

    if tokenized_corpus is not None and metadata_rows is not None: 
        print("Loaded existing tokenized corpus and metadata rows")
        return tokenized_corpus, metadata_rows
    
    # if not returned, they are not already built, so build the tokenized corpus and metadata rows
    tokenized_corpus, metadata_rows = load_tokenized_corpus_and_metadata_in_chunks(
        corpus_path = corpus_path,
        metadata_cols = META_COLS,
        chunk_size = chunk_size,
        max_rows = max_rows
    )

    # save the built tokenized corpus and metadata rows
    save_pickle(tokenized_corpus, tokenized_path)
    print(f"Saved tokenized corpus to: {tokenized_path}")

    save_pickle(metadata_rows, metadata_path)
    print(f"Saved metadata rows to: {metadata_path}")

    return tokenized_corpus, metadata_rows
    
def load_or_build_bm25(
    tokenized_corpus: list[list[str]],
    bm25_path: str = BM25_PATH
):
    """
    Load BM25 index if it already exists. Otherwise, build the index and save it.
    Got syntax help from Lecture 5 Notes: Comparison between BM25 and embedding-based search section
    """
    bm25 = load_pickle_if_valid(bm25_path)

    if bm25 is not None:
        print("Loaded existing BM25 index.")
        return bm25

    # build and persist the bm25 index if the path does not already exist
    print("Building BM25 index from scratch...")

    # build the BM25 index object
    bm25 = BM25Okapi(tokenized_corpus)
    save_pickle(bm25, bm25_path)
    print("Saved BM25 index")

    return bm25

def load_or_build_search_artifacts(
    corpus_path: str,
    tokenized_path: str, 
    metadata_path: str,
    bm25_path: str,
    chunk_size: int,
    max_rows: int | None = None
):
    '''
    If BM25 index and metadata rows already exist, load them right without reading in the tokenized_path or corpus path.

    Fast path after BM25 index and metadata rows (for output display) have been persisted: Load existing BM25 index and metadata rows and return immediately.
    Slow path: build/load tokenized corpus and metadata, and then build/load BM25.
    '''
    bm25 = load_pickle_if_valid(bm25_path)
    metadata_rows = load_pickle_if_valid(metadata_path)

    if bm25 is not None and metadata_rows is not None:
        print("Loaded existing BM25 index and metadata rows...")
        return bm25, metadata_rows
    
    print("BM25 index and/or metadata_rows missing... Falling back to corpus artifacts")
    
    tokenized_corpus, metadata_rows = load_or_build_corpus_artifacts(
        corpus_path = corpus_path,
        tokenized_path = tokenized_path,
        metadata_path = metadata_path,
        chunk_size = chunk_size,
        max_rows = max_rows
    )

    bm25 = load_or_build_bm25(
        tokenized_corpus,
        bm25_path = bm25_path
    )

    # delete the tokenized_corpus from memory as we do not need it 
    del tokenized_corpus
    gc.collect()

    return bm25, metadata_rows

def bm25_search(query: str, bm25, metadata_rows: list[dict], top_k: int = 5) -> list[tuple[dict, float]]:
    """
    Return top_k documents ranked by BM25 score.
    Got syntax help from Lecture 5 notes: Comparison between BM25 and embedding-based search
    """
    tokenized_query = tokenize(query)
    scores = bm25.get_scores(tokenized_query)

    # argpartition finds the indices of the largest top k scores without fully sorting the entire scores array
    # after this step, the top k rows are in the last top k positions in the array
    top_k_indices = np.argpartition(scores, -top_k)[-top_k:]

    # sort those top k indices by their score values in descending order
    top_k_indices = top_k_indices[np.argsort(scores[top_k_indices])[::-1]]

    return [(metadata_rows[i], scores[i]) for i in top_k_indices] # returns ranked scores along with product metadata in descending order

# driver program
def search_products_bm25(query: str, top_k: int = 5, max_rows: int | None = None, verbose: bool = False):
    '''
    Run a BM25 search on the product/review corpus and return the top matching products.

    Loads or builds the BM25 index and metadata, then ranks products based on the query.
    Optionally limits the number of rows for faster testing.
    '''
    # add suffix at the filepaths indicating number of rows for testing if we test a smaller subset
    tokenized_path = (
        f"{DATA_DIR}/tokenized_corpus_{max_rows}.pkl"
        if max_rows is not None else TOKENIZED_PATH
    )

    metadata_path = (
        f"{DATA_DIR}/metadata_rows_{max_rows}.pkl"
        if max_rows is not None else METADATA_PATH
    )

    bm25_path = (
        f"{DATA_DIR}/bm25_index_{max_rows}.pkl"
        if max_rows is not None else BM25_PATH
    )

    if verbose: 
        corpus_len = get_total_rows(CORPUS_PATH, max_rows = max_rows)
        print(f"Loaded corpus length: {corpus_len}")

    bm25, metadata_rows = load_or_build_search_artifacts(
        corpus_path = CORPUS_PATH,
        tokenized_path = tokenized_path,
        metadata_path = metadata_path,
        bm25_path = bm25_path,
        chunk_size = CHUNK_SIZE,
        max_rows = max_rows
    )

    results = bm25_search(
        query = query,
        bm25 = bm25,
        metadata_rows = metadata_rows,
        top_k = top_k
    )

    # for downstream use: the metdata we want to print will be controlled by the UI, not this function itself
    if verbose: 
        print(f"\n======= QUERY =======: {query}")

        for rank, (row, score) in enumerate(results, start=1):
            print(f"\nRank {rank}")
            print("Score:", round(float(score), 4))
            print("ASIN:", row.get("parent_asin", "N/A"))
            print("Title:", row.get("product_title", "N/A"))
            print("Category:", row.get("main_category", "N/A"))
            print("Store:", row.get("store", "N/A"))
            print("Price:", row.get("price", "N/A"))
            print("Rating:", row.get("average_rating", "N/A"))
            print("Description:", row.get("description", "N/A"))
            print("Review snippet:", (row.get("review_text_200") or "")[:200])
    
    # if verbose is False, just return the results: (metadata_row, score)
    return results

if __name__ == "__main__":
    TEST_QUERY = "wireless noise cancelling headphones"
    # MAX_ROWS = 50_000
    search_products_bm25(TEST_QUERY, top_k = 5, verbose = True)
