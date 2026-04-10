'''
Implements a BM25 keyoword based retreival system using LangChain
Following docs were used for help: https://docs.langchain.com/oss/python/integrations/retrievers/bm25
'''
import numpy as np
from rank_bm25 import BM25Okapi
from utils import tokenize, get_total_rows, load_pickle_if_valid, save_pickle, load_texts_and_metadata_in_chunks

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

def build_tokenized_corpus(
    texts: list[str]
) -> list[list[str]]:
    '''
    Convert raw retrieval texts into tokenized corpus (nested list of strs) for BM25. 
    '''
    print("Tokenizing corpus for BM25...")
    return [tokenize(text) for text in texts] # tokenize() returns a list[str], so return a list[list[str]] to match format BM25 expects

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
    texts, metadata_rows = load_texts_and_metadata_in_chunks(
        corpus_path = corpus_path,
        metadata_cols = META_COLS,
        chunk_size = chunk_size,
        max_rows = max_rows
    )

    tokenized_corpus = build_tokenized_corpus(texts)

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

def bm25_search(query: str, bm25, metadata_rows: list[dict], top_k: int = 5) -> list[tuple[dict, float]]:
    """
    Return top_k documents ranked by BM25 score.
    Got syntax help from Lecture 5 notes: Comparison between BM25 and embedding-based search
    """
    tokenized_query = tokenize(query)
    scores = bm25.get_scores(tokenized_query)
    sorted_indices = np.argsort(scores)[::-1]
    top_k_indices = sorted_indices[:top_k]
    return [(metadata_rows[i], scores[i]) for i in top_k_indices]

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

    results = bm25_search(
        query = query,
        bm25 = bm25,
        metadata_rows = metadata_rows,
        top_k = 5
    )

    print(f"\n======= QUERY =======: {query}")

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
