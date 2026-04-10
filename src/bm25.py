'''
Implemenets a BM25 keyoword based retreival system using LangChain
Following docs were used for help: https://docs.langchain.com/oss/python/integrations/retrievers/bm25
'''
import re 
import os
import pickle
import numpy as np
import polars as pl
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from rank_bm25 import BM25Okapi

DATA_DIR = "../data/processed"
TOKENIZED_PATH = f"{DATA_DIR}/tokenized_corpus.pkl"
BM25_PATH = f"{DATA_DIR}/bm25_index.pkl"
DOCS_PATH = f"{DATA_DIR}/langchain_documents.pkl"
CORPUS_PATH = f"{DATA_DIR}/retrieval_corpus.parquet"

CHUNK_SIZE = 10000

# define the tokenizer (whitespace, lowercase, puctuation and stopword removal)
def tokenize(text: str) -> list[str]:
    '''
    Takes in text string and converts it to a tokenized list format. 
    Aplies lowercasing, punctuation removal, and stopword removal
    '''
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s-]", " ", text) # remove punctuation, emojis and special characters: got syntax help from: https://stackoverflow.com/questions/5843518/remove-all-special-characters-punctuation-and-spaces-from-string
    tokens = text.split() # split on whitespace
    tokens = [t for t in tokens if t not in ENGLISH_STOP_WORDS]
    
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

# text column used for BM25
texts = corpus_df["retrieval_text"].fill_null("").to_list()

# read the columns that we need from the Parquet file 
corpus_df = pl.scan_parquet(CORPUS_PATH).select([
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
    "retrieval_text"
])

# keep metadata rows for the results we output 
metadata_rows = corpus_df.select([
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
]).to_dicts() # creates a list of dictionaries that we can index to get the rows/indices with the largest bm25 score

# if the tokenized corpus or bm25 index have already been persisted, load them from the above filepaths 
if os.path.exists(TOKENIZED_PATH):
    print("Loading existing tokenized corpus...")

    with open(TOKENIZED_PATH, "rb") as f:
        tokenized_corpus = pickle.load(f)
else: 
    print("Building tokenized corpus from scratch...")

    # tokenize docuuments
    tokenized_corpus = [tokenize(doc) for doc in texts]

    # save tokenized corpus documents 
    with open(TOKENIZED_PATH, "wb") as f:
        pickle.dump(tokenized_corpus, f)
        print("Saved tokenized corpus")

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