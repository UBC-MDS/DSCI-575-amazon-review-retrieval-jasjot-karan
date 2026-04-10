'''
Implemenets a BM25 keyoword based retreival system using LangChain
Following docs were used for help: https://docs.langchain.com/oss/python/integrations/retrievers/bm25
'''
import re 
import gc
import os
import pickle
import numpy as np
import polars as pl
import math
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from rank_bm25 import BM25Okapi

DATA_DIR = "../data/processed"
TOKENIZED_PATH = f"{DATA_DIR}/tokenized_corpus.pkl"
BM25_PATH = f"{DATA_DIR}/bm25_index.pkl"
DOCS_PATH = f"{DATA_DIR}/langchain_documents.pkl"
CORPUS_PATH = f"{DATA_DIR}/retrieval_corpus.parquet"

CHUNK_SIZE = 5000

STOPWORDS = set(ENGLISH_STOP_WORDS)

# define the tokenizer (whitespace, lowercase, puctuation and stopword removal)
def tokenize(text: str) -> list[str]:
    '''
    Takes in text string and converts it to a tokenized list format. 
    Aplies lowercasing, punctuation removal, and stopword removal
    '''
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s-]", " ", text) # remove punctuation, emojis and special characters: got syntax help from: https://stackoverflow.com/questions/5843518/remove-all-special-characters-punctuation-and-spaces-from-string
    tokens = text.split() # split on whitespace
    tokens = [t for t in tokens if t not in STOPWORDS]
    
    return tokens

def bm25_search(query, top_k = 5):
    '''
    Return top_k documents ranked by BM25 score.
    Referenced Lecture 5: Comparison between BM25 and embedding-based search: https://pages.github.ubc.ca/mds-2025-26/DSCI_575_adv-mach-learn_students/lectures/notes/05_info-retrieval-intro-to-transformers.html
    '''
    # tokenize the query
    tokenized_query = tokenize(query)
    
    # find the bm25 scores for all documents for tokenized_query asked
    scores = bm25.get_scores(tokenized_query)

    # sort the indixes based on score in descending order
    sorted_indices = np.argsort(scores)[::-1] # argsrot sorts in ascending order to reverse the order

    # slice out the top k indices 
    top_k_indices = sorted_indices[:top_k]

    # return the documents with their scores 
    return [(metadata_rows[i], scores[i]) for i in top_k_indices]

# count rows lazily 
total_rows = (
    pl.scan_parquet(CORPUS_PATH)
    .select(pl.len())
    .collect()
    .item()
)

print(f"Loaded corpus length: {total_rows}")

# if the tokenized corpus or bm25 index have already been persisted, load them from the above filepaths 
if os.path.exists(TOKENIZED_PATH):
    print("Loading existing tokenized corpus...")

    with open(TOKENIZED_PATH, "rb") as f:
        tokenized_corpus = pickle.load(f)
else: 
    print("Building tokenized corpus from scratch in chunks...")

    # tokenize docuuments by chunk
    tokenized_corpus = []

    num_chunks = math.ceil(total_rows / CHUNK_SIZE)

    for chunk_idx in range(num_chunks):
        offset = chunk_idx * CHUNK_SIZE 

        chunk_df = (
            pl.scan_parquet(CORPUS_PATH)
            .select(["retrieval_text"])
            .slice(offset, CHUNK_SIZE)
            .collect()
        )

        texts = chunk_df["retrieval_text"].fill_null("").to_list()
        tokenized_chunk = [tokenize(doc) for doc in texts]
        
        tokenized_corpus.extend(tokenized_chunk)
        print(f"Processed chunk {chunk_idx + 1}/{num_chunks}")

        # free that chunks df object, tokens and text from memory after we have added it to the tokenized_corpus
        del chunk_df, texts, tokenized_chunk
        gc.collect() # collect memory with garbage collector

    # save tokenized corpus documents 
    with open(TOKENIZED_PATH, "wb") as f:
        pickle.dump(tokenized_corpus, f, protocol = pickle.HIGHEST_PROTOCOL)
        print("Saved tokenized corpus")

# build the metadata rows in chunks too 
print("Building metadata rows in chunks...")
metadata_rows = []
num_chunks = math.ceil(total_rows / CHUNK_SIZE)

for chunk_idx in range(num_chunks):
    offset = chunk_idx * CHUNK_SIZE 

    chunk_meta_df = (
        pl.scan_parquet(CORPUS_PATH)
        .select([
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
            "all_review_titles"
        ])
        .slice(offset, CHUNK_SIZE)
        .collect()
    )

    metadata_rows.extend(chunk_meta_df.to_dicts())

    print(f"Processed metadata chunk {chunk_idx + 1}/{num_chunks}")

    del chunk_meta_df
    gc.collect()

if os.path.exists(BM25_PATH):
    print("Loading existing BM25 index...")

    with open(BM25_PATH, "rb") as f:
        bm25 = pickle.load(f)
else:
    # build and persist the bm25 index 
    print("Building BM25 index from scratch...")

    # build the BM25 index object
    bm25 = BM25Okapi(tokenized_corpus)

    with open(BM25_PATH, "wb") as f:
        pickle.dump(bm25, f)
        print("Saved BM25 index")

query = "wireless noise cancelling headphones"

results = bm25_search(query, top_k = 5)

for rank, (row, score) in enumerate(results, start = 1):
    print(f"\nRank {rank}")
    print("Score: ", round(float(score), 4))
    print("ASIN:", row.get("parent_asin"))
    print("Title:", row.get("product_title"))
    print("Category:", row.get("main_category"))
    print("Store:", row.get("store"))
    print("Price:", row.get("price"))
    print("Rating:", row.get("average_rating"))
    print("Description:",row.get("description"))