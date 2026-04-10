import re
import os
import numpy as np
import polars as pl
import pickle
import polars as pl
import math
import gc
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

STOPWORDS = set(ENGLISH_STOP_WORDS)

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
        print(f"Corrupted pickle detected at {path}. Rebuilding...")
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

def load_texts_and_metadata_in_chunks(
    corpus_path: str,
    metadata_cols: list[str],
    chunk_size: int,
    max_rows: int | None = None
) -> tuple[list[list[str]], list[dict]]:
    '''
    Returns retrieval text (for BM25 search and embeddings) and metadata rows in chunks
    '''
    print(f"Loading retrival text and metadata rows from: {corpus_path}")

    total_rows = get_total_rows(corpus_path, max_rows = max_rows)
    num_chunks = math.ceil(total_rows / chunk_size)

    texts: list[str] = []
    metadata_rows: list[dict] = []

    corpus_lf = pl.scan_parquet(corpus_path).slice(0, total_rows)

    chunk_lf = corpus_lf.select(
        [pl.col("retrieval_text")] + [pl.col(col) for col in metadata_cols]
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
        texts.extend(chunk_df["retrieval_text"].to_list())

        # turns chunk into a list of dictionaries and add each dict to metadata_rows
        metadata_rows.extend(chunk_df.select(metadata_cols).to_dicts())
                             
        print(f"Processed chunk {chunk_idx + 1}/{num_chunks}")

        # free that chunnk df object from memory after we have added it to the tokenized_corpus so memory does not keep growing chunk by chunk
        del chunk_df
        gc.collect() 

    return texts, metadata_rows