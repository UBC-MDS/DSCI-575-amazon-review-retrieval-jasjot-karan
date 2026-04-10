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

from utils import tokenize, polars_tokenize_expr, bm25_search, STOPWORDS

DATA_DIR = "../data/processed"
TOKENIZED_PATH = f"{DATA_DIR}/tokenized_corpus.pkl"
BM25_PATH = f"{DATA_DIR}/bm25_index.pkl"
CORPUS_PATH = f"{DATA_DIR}/retrieval_corpus.parquet"

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

def load_or_build_tokenized_corpus(
    corpus_path: str,
    tokenized_path: str = TOKENIZED_PATH,
    chunk_size: int = CHUNK_SIZE,
    max_rows: int | None = None
) -> list[list[str]]:
    '''
    Load tokenized corpus if it already exists and filepath is valid.
    Otherwise build it in chunks and save it.
    '''
    if os.path.exists(tokenized_path) and os.path.getsize(tokenized_path) > 0:
        print("Loading existing tokenized corpus...")
        try:
            with open(tokenized_path, "rb") as f:
                tokenized_corpus = pickle.load(f)
            return tokenized_corpus
        except (EOFError, pickle.UnpicklingError):
            print("Tokenized corpus file is corrupted")
            os.remove(tokenized_path)

    
    print("Building tokenized corpus from scratch in chunks...")

    total_rows = get_total_rows(corpus_path, max_rows = max_rows)

    num_chunks = math.ceil(total_rows / chunk_size)

    # tokenize docuuments by chunk
    tokenized_corpus = []

    # define the lazyframes outside of the loop so they are not re-iterated over
    corpus_lf = pl.scan_parquet(corpus_path)
    tokens_lf = corpus_lf.select([polars_tokenize_expr("retrieval_text")])

    for chunk_idx in range(num_chunks):
        offset = chunk_idx * chunk_size 

        chunk_df = (
            tokens_lf
            .slice(offset, chunk_size)
            .collect()
        )

        # add the current chunk to our existing tokenized_corpus
        tokenized_chunk = chunk_df["tokens"].to_list()
        tokenized_corpus.extend(tokenized_chunk)

        print(f"Processed chunk {chunk_idx + 1}/{num_chunks}")

        # free that chunks df object, tokens and text from memory after we have added it to the tokenized_corpus
        del chunk_df, tokenized_chunk
        gc.collect() # collect memory with garbage collector

    # save tokenized corpus documents 
    with open(tokenized_path, "wb") as f:
        pickle.dump(tokenized_corpus, f)  
    print("Saved tokenized corpus")

    return tokenized_corpus

def load_metadata_rows(
    corpus_path: str,
    chunk_size: int = CHUNK_SIZE,
    max_rows: int | None = None
) -> list[dict]:
    """
    Load metadata rows in chunks
    """
    # build the metadata rows in chunks too 
    print("Building metadata rows in chunks...")
    
    total_rows = get_total_rows(corpus_path, max_rows = max_rows)
    num_chunks = math.ceil(total_rows / chunk_size)

    metadata_rows = []

    corpus_lf = pl.scan_parquet(corpus_path)
    metadata_lf = corpus_lf.select(META_COLS)
    
    for chunk_idx in range(num_chunks):
        offset = chunk_idx * chunk_size 

        chunk_meta_df = (
            metadata_lf
            .slice(offset, chunk_size)
            .collect()
        )

        metadata_rows.extend(chunk_meta_df.to_dicts())

        print(f"Processed metadata chunk {chunk_idx + 1}/{num_chunks}")

        # free the memory used by the chunk
        del chunk_meta_df
        gc.collect()

    return metadata_rows

def load_or_build_bm25(
    tokenized_corpus: list[list[str]],
    bm25_path: str = BM25_PATH
):
    """
    Load BM25 index if it already exists. Otherwise, build the index and save it.
    Got syntax help from Lecture 5 Notes: Comparison between BM25 and embedding-based search section
    """
    if os.path.exists(bm25_path):
        print("Loading existing BM25 index...")

        try:
            with open(bm25_path, "rb") as f:
                bm25 = pickle.load(f)
            return bm25
        except (EOFError, pickle.UnpicklingError):
            print("BM25 index file is corrupted. Please rebuild it.")
            os.remove(bm25_path)

    # build and persist the bm25 index if the path does not already exist
    print("Building BM25 index from scratch...")

    # build the BM25 index object
    bm25 = BM25Okapi(tokenized_corpus)

    with open(bm25_path, "wb") as f:
        pickle.dump(bm25, f)
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
        if max_rows is not None else f"{DATA_DIR}/tokenized_corpus.pkl"
    )

    bm25_path = (
        f"{DATA_DIR}/bm25_index_{max_rows}.pkl"
        if max_rows is not None else f"{DATA_DIR}/bm25_index.pkl"
    )

    corpus_len = get_total_rows(CORPUS_PATH, max_rows = max_rows)
    print(f"Loaded corpus length: {corpus_len}")

    tokenized_corpus = load_or_build_tokenized_corpus(
        corpus_path = CORPUS_PATH,
        tokenized_path = tokenized_path,
        chunk_size = CHUNK_SIZE,
        max_rows = max_rows
    )

    metadata_rows = load_metadata_rows(
        corpus_path = CORPUS_PATH,
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
    MAX_ROWS = 20000
    main(TEST_QUERY, max_rows = MAX_ROWS)
