'''
Implemenets a BM25 keyoword based retreival system using LangChain
Following docs were used for help: https://docs.langchain.com/oss/python/integrations/retrievers/bm25
'''
import gc
import os
import pickle
import polars as pl
import math
from rank_bm25 import BM25Okapi

from utils import polars_tokenize_expr, bm25_search

DATA_DIR = "../data/processed"
TOKENIZED_PATH = f"{DATA_DIR}/tokenized_corpus.pkl"
BM25_PATH = f"{DATA_DIR}/bm25_index.pkl"
CORPUS_PATH = f"{DATA_DIR}/retrieval_corpus.parquet"
METADATA_PATH = f"{DATA_DIR}/metadata_rows.pkl"

CHUNK_SIZE = 10000

META_COLS = [
    "parent_asin",
    "product_title",
    "description",
    "main_category",
    "store",
    "price",
    "average_rating",
    "rating_number",
    "review_count",
    "features",
    "categories",
    "all_review_titles",
]

def get_total_rows(corpus_path: str, max_rows: int | None = None) -> int: 
    """
    Count rows lazily from the pqrquet file 
    """
    total_rows = (
        pl.scan_parquet(corpus_path)
        .select(pl.len())
        .collect()
        .item()
    )

    if max_rows is not None: 
        return min(total_rows, max_rows)
    
    return total_rows

def load_corpus_in_chunks(
    corpus_path: str,
    chunk_size: int = CHUNK_SIZE,
    max_rows: int | None = None
) -> tuple[list[list[str]], list[dict]]:
    '''
    Build tokenized corpus and metadata rows.
    Returns tokenized corpus (nested list holding strs) and metadata rows (list of dicts)
    '''
    print("Building tokenized corpus and metadata rows...")

    total_rows = get_total_rows(corpus_path, max_rows = max_rows)
    num_chunks = math.ceil(total_rows / chunk_size)

    tokenized_corpus: list[list[str]] = [] # BM25 requires nested list
    metadata_rows: list[dict] = []

    corpus_lf = pl.scan_parquet(corpus_path).slice(0, total_rows)
    chunk_lf = corpus_lf.select(
        [polars_tokenize_expr("retrieval_text")] + [pl.col(col) for col in META_COLS]
    )

    for chunk_idx in range(num_chunks):
        offset = chunk_idx * chunk_size 

        # slice out the current block/chunk we are at
        chunk_df = (
            chunk_lf
            .slice(offset, chunk_size)
            .collect()
        )

        # add the current chunk's tokens and metadata to our existing tokenized_corpus and metadata_rows 
        tokenized_corpus.extend(chunk_df["tokens"].to_list())
        metadata_rows.extend(chunk_df.select(META_COLS).to_dicts())
                             
        print(f"Processed chunk {chunk_idx + 1}/{num_chunks}")

        # free that chunnk df object from memory after we have added it to the tokenized_corpus so memory does not keep growing chunk by chunk
        del chunk_df
        gc.collect() 

    return tokenized_corpus, metadata_rows

def load_pickle_if_valid(path: str): 
    '''
    Load a pickle file if it exists and is not corrupted, otherwise it returns None.
    '''
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return None
    
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
        
    except (EOFError, pickle.UnpicklingError):
        print(f"Corrupted pickle detected at {path}. Rebuilding...")
        os.remove(path)

def save_pickle(obj, path: str) -> None:
    '''
    Saves object to pickle
    '''
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def load_or_build_corpus_artifacts(
    corpus_path: str,
    tokenized_path: str = TOKENIZED_PATH,
    metadata_path: str = METADATA_PATH,
    chunk_size: int = CHUNK_SIZE,
    max_rows: int | None = None
) -> tuple[list[list[str]], list[dict]]:
    '''
    Load tokenized corpus and metadata rows if they already exist. Else, build both by chunking.
    '''
    tokenized_corpus = load_pickle_if_valid(tokenized_path)
    metadata_rows = load_pickle_if_valid(metadata_path)

    if tokenized_corpus is not None and metadata_rows is not None: 
        print("Loaded existing tokenized corpus and metadata rows")
        return tokenized_corpus, metadata_rows
    
    # if not returned, they are not already built, so build the tokenized corpus and metadata rows
    tokenized_corpus, metadata_rows = load_corpus_in_chunks(
        corpus_path = corpus_path,
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

def run_bm25_query(
    query: str,
    bm25,
    metadata_rows: list[dict],
    top_k: int = 5
) -> list[tuple[dict, float]]: # tuple of metadata row as a dictionary and float (bm25 score)
    """
    Run one BM25 query and return ranked results.
    """
    results = bm25_search(
        query = query, 
        bm25 = bm25,
        metadata_rows = metadata_rows,
        top_k = top_k
    )

    return results

# driver program
def main(query: str, max_rows: int | None = None):
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

    corpus_len = get_total_rows(CORPUS_PATH, max_rows = max_rows)
    print(f"Loaded corpus length: {corpus_len}")

    tokenized_corpus, metadata_rows = load_or_build_corpus_artifacts(
        corpus_path = CORPUS_PATH,
        tokenized_path = tokenized_path,
        metadata_path = metadata_path,
        chunk_size = CHUNK_SIZE,
        max_rows = max_rows
    )

    bm25 = load_or_build_bm25(
        tokenized_corpus = tokenized_corpus,
        bm25_path = bm25_path
    )

    results = run_bm25_query(
        query = query,
        bm25 = bm25,
        metadata_rows = metadata_rows,
        top_k = 5
    )

    for rank, (row, score) in enumerate(results, start=1):
        print(f"\nRank {rank}")
        print("Score:", round(float(score), 4))
        print("ASIN:", row.get("parent_asin"))
        print("Title:", row.get("product_title"))
        print("Category:", row.get("main_category"))
        print("Store:", row.get("store"))
        print("Price:", row.get("price"))
        print("Rating:", row.get("average_rating"))
        print("Description:", row.get("description"))

if __name__ == "__main__":
    TEST_QUERY = "wireless noise cancelling headphones"
    MAX_ROWS = 50_000
    main(TEST_QUERY, max_rows = MAX_ROWS)
