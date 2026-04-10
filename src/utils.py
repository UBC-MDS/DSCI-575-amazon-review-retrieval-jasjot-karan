import re
import numpy as np
import polars as pl
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

STOPWORDS = set(ENGLISH_STOP_WORDS)

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
    Return a Polars expression that tokenizes a text column into list[str].
    Useful for corpus preprocessing in chunks.
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