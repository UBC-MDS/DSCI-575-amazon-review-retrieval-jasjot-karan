'''
Implements hybrid retrieval for the Amazon product search application by
combining BM25 keyword search and semantic vector search using Reciprocal
Rank Fusion (RRF).
'''
from collections import defaultdict
from src.bm25 import bm25_search
from src.semantic import semantic_search

def hybrid_search(
    query: str, 
    bm25_index,
    bm25_metadata_rows: list[dict],
    faiss_index,
    semantic_metadata_rows: list[dict],
    model,
    top_k: int = 5,
    candidate_multiplier: int = 3, # retrieves more than top k from each method before fusing the scores so the hybrid score has a better signal ex) If one method ranks a doc 2 and the otehr ranks it 12, having no candidate_multiplier results in only top 5 rankings compared compared, which could miss some similarities
    rrf_k: int = 60 # RRF smoothing constant
) -> list[tuple[dict, float]]:
    '''
    Run hybrid retrieval using Reciprocal Rank Fusion (RRF).

    BM25 and semantic scores live on different numeric scales so RRF combines ranks of documents/products for each retrieval method instead.
    So documents that are ranked highly by both retrievers get a high score/reward. 

    Returns the same tuple of (metadata_row, hybrid_score). 

    Referenced lecture notes 5 for the hybrid search idea of RRF and hyperparameter k: https://pages.github.ubc.ca/mds-2025-26/DSCI_575_adv-mach-learn_students/lectures/notes/05_info-retrieval-intro-to-transformers.html/
    Also referecned the code from the Reciprocal Rank Fusion section from Mikhail Berkov's Medium Article from August 14, 2025: https://medium.com/thinking-sand/hybrid-search-with-bm25-and-rank-fusion-for-accurate-results-456a70305dc5
    '''
    # number of candidate results that can be fused for each method
    candidate_k = candidate_multiplier * top_k

    # get the largest candidate pools from both retrievers
    bm25_results = bm25_search(
        query = query,
        bm25 = bm25_index,
        metadata_rows = bm25_metadata_rows,
        top_k = candidate_k
    )

    semantic_results = semantic_search(
        query = query,
        index = faiss_index,
        metadata_rows = semantic_metadata_rows ,
        model = model,
        top_k = candidate_k
    )

    # default dict to store the hybrid scores
    hybrid_scores = defaultdict(float) # key -> parent_asin: value -> hybrid_score
    row_lookup = defaultdict(dict)

    # add the bm25 ranks to the hybrid score bsaed on the query 
    for rank, (metadata_row, _) in enumerate(bm25_results, start = 1):
        document_id = metadata_row.get('parent_asin')
        if document_id is None:
            continue

        # compute the reciprocal rank fusion score: 1 / (k + rank of bm25 retrieval on document)
        hybrid_scores[document_id] += 1 / (rrf_k + rank)
        row_lookup[document_id] = metadata_row

    # add the semantic rankings to the hybrid scores for each document that also appears in the semantic retrieval
    for rank, (metadata_row, _) in enumerate(semantic_results, start = 1):
        document_id = metadata_row.get('parent_asin')
        if document_id is None:
            continue 
        
        # compute RRF score for each top k document for semantic retrieval, and add the semantic RRF score to each respectve document's hybrid score
        hybrid_scores[document_id] += 1 / (rrf_k + rank)
        row_lookup[document_id] = metadata_row
    
    # sort the documents by hybrid score in descending order
    ranked_doc_ids = sorted(
        hybrid_scores.items(),
        key = lambda x: x[1], # sort by hybrid score in desc order
        reverse = True
    )

    final_results = [
        (row_lookup[document_id], hybrid_score)
        for document_id, hybrid_score in ranked_doc_ids[:top_k]
    ]

    return final_results 




