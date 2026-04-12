'''
Semantic Search

A semantic retrieval system with two steps:

Embeddings: Using sentence-transformers to generate embeddings for our text documents. 
We used all-MiniLM-L6-v2 model. 

Indexing: Using FAISS (Facebook Similarity Search) to build the index and later do the search. We persist the FAISS index.
'''
import gc
import glob
import faiss
import math
import numpy as np 
import polars as pl
from pathlib import Path
from sentence_transformers import SentenceTransformer

from utils import get_total_rows, load_pickle_if_valid, save_pickle, META_COLS

# make sure the data can be processed regardless of the directory we are in
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data/processed"

CORPUS_PATH = DATA_DIR / "retrieval_corpus.parquet"
FAISS_INDEX_PATH = DATA_DIR / "faiss_index.index"
METADATA_PATH = DATA_DIR / "metadata_rows.pkl"
EMBEDDINGS_PATH = DATA_DIR / "embeddings.npy"

MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 10_000
EMBED_BATCH_SIZE = 256 # vector size to send to FAISS to embed each batch

def load_sentence_transformer_smodel(model_name: str = MODEL_NAME) -> SentenceTransformer:
    '''
    Load the sentence transformers model to be used for embedding and semantic search. 
    '''
    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    return model 

def get_embedding_chunk_pattern(max_rows: int | None = None):
    '''
    Return the chunk file pattern for saved embedding chunks.
    '''
    if max_rows is not None:
        return f"embeddings_{max_rows}_chunk_*.npy"
    return "embeddings_chunk_*.npy"

def get_embeddings_chunk_path(chunk_idx: int, max_rows: int | None = None) -> Path:
    '''
    Return the filepath for one embedding chunk.
    '''
    if max_rows is not None:
        return DATA_DIR / f"embeddings_{max_rows}_chunk_{chunk_idx}.npy"

    return DATA_DIR / f"embeddings_chunk_{chunk_idx}.npy"

def load_embedding_chunk_files(max_rows: int | None = None) -> list[str]:
    '''
    Return sorted embedding chunk file paths.
    '''
    chunk_pattern = get_embedding_chunk_pattern(max_rows = max_rows)
    return sorted(DATA_DIR.glob(chunk_pattern))

def build_faiss_index_from_embedding_chunks(
    chunk_files: list[str],
    index_path: str = FAISS_INDEX_PATH,
    M: int = 32
):
    """
    Rebuild the FAISS index from the persisted embedding chunks (if they exist).
    
    We persist embeddings chunk by chunk instead of keeping the full embedding
    matrix in memory. This is safer for bigger datasets and also lets us rebuild
    the FAISS index later without embedding the entire corpus again.
    """
    if not chunk_files:
        raise FileNotFoundError("No embedding chunk files found")

    index = None 

    for chunk_file in chunk_files: 
        embeddings = np.load(chunk_file).astype("float32")

        # rebuild the FAISS index if the persisted embeddings exist
        if index is None: 
            embedding_dim = embeddings.shape[1]
            index = faiss.IndexHNSWFlat(
                embedding_dim,
                M,
                faiss.METRIC_INNER_PRODUCT
            )

        index.add(embeddings)
    
    # persist the index if we built it
    faiss.write_index(index, str(index_path))
    print(f"Saved rebuilt FAISS index to: {index_path}")

    return index

def build_faiss_index_and_metadata(
    corpus_path: str, 
    index_path: str = FAISS_INDEX_PATH,
    metadata_path: str = METADATA_PATH,
    chunk_size: int = CHUNK_SIZE,
    batch_size: int = EMBED_BATCH_SIZE,
    max_rows: int | None = None,
    M: int = 32, # graph connectivivty to each neighbour -> higher means better recall and more memory but is slower 
):
    """
    Build the FAISS index and semantic embeddings of the metadata and persist both locally. 
    """
    model = load_sentence_transformer_smodel()

    total_rows = get_total_rows(corpus_path, max_rows = max_rows)
    print(f"Loaded corpus length: {total_rows}")

    metadata_rows: list[dict] = []
    index = None

    corpus_lf = (
        pl.scan_parquet(corpus_path)
        .slice(0, total_rows)
        .select(["retrieval_text"] + META_COLS)
    )

    num_chunks = math.ceil(total_rows / chunk_size)

    for chunk_idx in range(num_chunks): 
        # slice out the current block/chunk we are at 
        offset = chunk_idx * chunk_size

        chunk_df = (
            corpus_lf
            .slice(offset, chunk_size) # start at row offset and take next chunk_size rows
            .collect()
        ) 

        texts = chunk_df["retrieval_text"].fill_null("").to_list()
        chunk_metadata = chunk_df.select(META_COLS).to_dicts()

        # embed the retrieval text using our model
        # got syntax help and learned extra params such as convert_to_numpy from: https://sbert.net/docs/package_reference/sentence_transformer/model.html
        embeddings = model.encode(
            texts,
            batch_size = batch_size,
            show_progress_bar = True,
            convert_to_numpy = True, # for easier storage downstream
            normalize_embeddings = True # normalize our embeddings for cosine similarity/FAISS so vector magnitutude does not skew similarity results
        ).astype("float32")

        # persist the embeddings from each chunk 
        chunk_path = get_embeddings_chunk_path(chunk_idx = chunk_idx, max_rows = max_rows)
        np.save(chunk_path, embeddings)
        print(f"Saved embedding chunk to: {chunk_path}")

        if index is None: 
            embedding_dim = embeddings.shape[1]
            # chose to use HNSW model as it does not compare the query across all embeddings like IndexFlatIP which becomes very slow
            # HNSW uses an approximate nearest neighbour approach where it navigates through promising neighbours of top results only which makes semantic search must faster, while also maintaining a solid recall. 
            # Is recommended by FAISS as a good speed/accuracy trade off for larger datasets: 
            # used the following to help with justification, but chose HNSW due to its ANN approach to save speed: https://www.pinecone.io/learn/series/faiss/hnsw/ and https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index
            index = faiss.IndexHNSWFlat(
                embedding_dim, 
                M,
                faiss.METRIC_INNER_PRODUCT # this computes cosine similarity according to the docs (normalized embeddings + inner product): https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances
            )
        
        # add the relevant embeddings to the IndexHNSWFlat using a cosine similarity approach to build the index
        # referenced: https://www.pinecone.io/learn/series/faiss/hnsw/
        index.add(embeddings)

        # add the metadata for this chunk (one dict per row) to metadata_rows (each dict is on its own line)
        # order has to matxh index.add(embeddings) as every metadata row has the same index as its embedding in the FAISS index
        metadata_rows.extend(chunk_metadata)

        print(f"Processed chunk: {chunk_idx + 1}/{num_chunks}")

        # delete and free memory for the embeddings we just added to metadata_rows
        del chunk_df, texts, chunk_metadata, embeddings
        gc.collect()
    
    # persist the index 
    faiss.write_index(index, str(index_path))
    print(f"Saved FAISS index to: {index_path}")

    save_pickle(metadata_rows, metadata_path)
    print(f"Saved Semantic metadata rows to: {metadata_path}")

    return index, metadata_rows

def load_or_build_semantic_artifacts(
    corpus_path: str,
    index_path: str = FAISS_INDEX_PATH,
    metadata_path: str = METADATA_PATH,
    chunk_size: int = CHUNK_SIZE,
    batch_size: int = EMBED_BATCH_SIZE,
    max_rows: int | None = None
): 
    """
    Load semantic artifacts if they already exist. 
    Otherwise, rebuild them from the saved embeddings or from the corpus.
    """

    # try to load the metadata rows (should exist from previous step)
    metadata_rows = load_pickle_if_valid(metadata_path)

    try:
        index = faiss.read_index(str(index_path))
    except Exception: 
        index = None 
    
    chunk_files = load_embedding_chunk_files(max_rows = max_rows)

    if index is not None and metadata_rows is not None: 
        print("Loaded existing FAISS index and semantic metadata...")
        return index, metadata_rows

    # case 2: if index is missing but metadata_rows and chunked embeddings are not missing: build the index from the persisted embeddings
    if index is None and metadata_rows is not None and chunk_files: 
        print("FAISS index missing, but metadata and embedding chunks exist. Rebuilding index from embedding chunks...")
        
        index = build_faiss_index_from_embedding_chunks(
            chunk_files = chunk_files,
            index_path = index_path
        )

        return index, metadata_rows

    print("FAISS index and/or semantic metadata rows and embeddings are missing. Building from corpus...")

    index, metadata_rows =  build_faiss_index_and_metadata(
        corpus_path = corpus_path, 
        index_path = index_path,
        metadata_path = metadata_path,
        chunk_size = chunk_size,
        batch_size = batch_size,
        max_rows = max_rows
    )

    return index, metadata_rows

def semantic_search(
    query: str,
    index,
    metadata_rows: list[dict],
    model: SentenceTransformer,
    top_k: int = 5
) -> list[tuple[dict, float]]: # returns metadata row and score
    """
    Returns top k semantic retrieval results using FAISS.
    Used lecture 5 notes, Comparison between BM25 and embedding-based search section for syntax help
    """
    query_embedding = model.encode(
        [query],
        convert_to_numpy = True,
        normalize_embeddings = True # normalize for cosine similarity
    ).astype("float32")
    
    # search the index for the top 5 most relevant embeddings
    # referenced: https://www.pinecone.io/learn/series/faiss/hnsw/ for implementation help
    scores, indices = index.search(query_embedding, top_k)

    results = []
    for idx, score in zip(indices[0], scores[0]):
        # skip the index if it is not relevant
        if idx == -1: 
            continue
        results.append((metadata_rows[idx], float(score)))
    
    return results

def run_semantic_search(
    query: str, 
    top_k: int = 5,
    max_rows: int | None = None, 
    verbose: bool = False
):
    index_path = (
        f"{DATA_DIR}/faiss_index_{max_rows}.index"
        if max_rows is not None else FAISS_INDEX_PATH
    )

    metadata_path = (
        f"{DATA_DIR}/metadata_rows_{max_rows}.pkl"
        if max_rows is not None else METADATA_PATH
    )

    index, metadata_rows = load_or_build_semantic_artifacts(
        corpus_path = CORPUS_PATH,
        index_path = index_path,
        metadata_path = metadata_path,
        chunk_size = CHUNK_SIZE,
        batch_size = EMBED_BATCH_SIZE,
        max_rows = max_rows
    )

    model = load_sentence_transformer_smodel()

    results = semantic_search(
        query = query,
        index = index,
        metadata_rows = metadata_rows,
        model = model,
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
            print("Review snippets:", (row.get("review_text_200") or "")[:200])
    
    # if verbose is False, just return the results: (metadata_row, score)
    return results

if __name__ == "__main__":
    TEST_QUERY = "wireless noise cancelling headphones"
    # MAX_ROWS = 50_000
    run_semantic_search(TEST_QUERY, top_k = 5, verbose = True)
    