import re
import os
import numpy as np
import polars as pl
import pickle
import math
import gc
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from tqdm import tqdm # for progress bar

STOPWORDS = set(ENGLISH_STOP_WORDS)

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
    "review_text_200"
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
        print(f"Pickle obj does not exist or corrupted pickle detected at {path}. Rebuilding...")
        os.remove(path)
 
def save_pickle(obj, path: str) -> None:
    '''
    Saves object to pickle
    '''
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def tokenize(text: str) -> list[str]:
    """
    Python tokenizer for short strings like user queries.
    """
    if text is None:
        return []

    text = text.lower()
    text = re.sub(r"[^a-z0-9\s-]", " ", text)
    tokens = text.split()
    tokens = [t for t in tokens if t and t not in STOPWORDS]
    return tokens

def polars_tokenize_expr(text_col: str = "retrieval_text") -> pl.Expr:
    """
    Return a Polars expression that tokenizes a text column into list[str] by removing symbols, punctuationm and stopwords.
    Useful for vectorized and optimized corpus preprocessing in chunks instead of Python's per row interpretation and no optimizer.
    Used below sources for help:

        https://stackoverflow.com/questions/77202907/slow-performance-of-python-polars-in-applying-pl-element-filter
        https://docs.pola.rs/api/python/stable/reference/expressions/api/polars.element.html
        https://github.com/pola-rs/polars/issues/13772
    """
    return (
        pl.col(text_col)
        .fill_null("")
        .str.to_lowercase()
        .str.replace_all(r"[^a-z0-9\s-]", " ")
        .str.split(" ")
        .list.eval(
            pl.element().filter(
                (pl.element() != "") &
                (~pl.element().is_in(STOPWORDS))
            )
        )
        .alias("tokens")
    )

def load_tokenized_corpus_and_metadata_in_chunks(
    corpus_path: str,
    metadata_cols: list[str],
    chunk_size: int,
    max_rows: int | None = None
) -> tuple[list[list[str]], list[dict]]:
    '''
    Loads retrieval text and metadata from parquet in chunks, tokenizes the text,
    and returns the full tokenized corpus plus metadata rows
    '''
    print(f"Loading and tokenizing retrieval text from: {corpus_path}")

    total_rows = get_total_rows(corpus_path, max_rows = max_rows)
    num_chunks = math.ceil(total_rows / chunk_size)

    tokenized_corpus: list[list[str]] = []
    metadata_rows: list[dict] = []

    corpus_lf = pl.scan_parquet(corpus_path).slice(0, total_rows)

    chunk_lf = corpus_lf.select(
        [polars_tokenize_expr("retrieval_text")] + [pl.col(col) for col in metadata_cols]
    )

    for chunk_idx in tqdm(range(num_chunks), desc = "Tokenizing chunks", unit = "chunk"):
        offset = chunk_idx * chunk_size 

        # slice out the current block/chunk we are at
        chunk_df = (
            chunk_lf
            .slice(offset, chunk_size)
            .collect()
        )

        # add the current chunk's tokens and metadata to our existing tokenized_corpus and metadata_rows 
        tokenized_corpus.extend(chunk_df["tokens"].to_list())

        # turns chunk into a list of dictionaries and add each dict to metadata_rows on a new line, preserving the index
        metadata_rows.extend(chunk_df.select(metadata_cols).to_dicts())
                             
        print(f"Processed chunk {chunk_idx + 1}/{num_chunks}")

        # free that chunnk df object from memory after we have added it to the tokenized_corpus so memory does not keep growing chunk by chunk
        del chunk_df
        gc.collect()

    return tokenized_corpus, metadata_rows