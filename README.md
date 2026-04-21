# Amazon Electronics Product Retrieval

A production-style information retrieval system built on Amazon Electronics review data. The system implements and compares three retrieval strategies: **BM25 keyword search**, **semantic vector search**, and a **hybrid approach** using Reciprocal Rank Fusion (RRF). The application surfaces results through a clean Streamlit UI modeled after the Amazon experience. Users can search, compare methods side-by-side, and provide thumbs up/down relevance feedback that gets persisted for evaluation.

[![Python](https://img.shields.io/badge/python-3.12-blue)](https://www.python.org/) [![Streamlit](https://img.shields.io/badge/streamlit-1.36.0-red)](https://streamlit.io/) [![FAISS](https://img.shields.io/badge/faiss--cpu-1.8.*-green)](https://github.com/facebookresearch/faiss) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Table of Contents
 
- [RAG Pipeline](#rag-pipeline-workflow)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Data Processing](#data-processing)
- [Retrieval Workflows](#retrieval-workflows)
- [Environment Setup](#environment-setup)
- [Reproducing the Results](#reproducing-the-results)
- [Usage Examples](#usage-examples)
- [User Feedback & Evaluation](#user-feedback--evaluation)
- [Authors](#authors)
---

## RAG Pipeline Workflow 

The pipeline has three stages: **Retrieval** (semantic similarity search over FAISS), **Context & Prompt Construction** (formatting retrieved products into a structured prompt for the LLM), and **Generation** (grounded answer from phi4-mini via Ollama).

```mermaid
flowchart TD
    A[User Query] --> B[all-MiniLM-L6-v2<br/>Sentence Transformer]
    B --> C[Normalized Query Embedding<br/>384-dim vector]
    
    C --> D[FAISS HNSW Index<br/>Inner Product / Cosine Similarity]
    D --> E[Top-K Product Indices<br/>+ Similarity Scores]
    
    E --> F[Metadata Lookup<br/>metadata_rows dict]
    F --> G[Top-K Retrieved Products<br/>with product metadata + similarity scores]
    
    G --> H[build_context<br/>combine top k products into a string]
    H --> I[build_prompt<br/>system prompt + product context + user query]
    
    I --> J[phi4-mini LLM model via Ollama<br/>temperature=0.0]
    J --> K[Grounded Answer with ASIN Citations]
    
    G --> L["Final Output Dict<br/>(Query + LLM Answer + Retrieved Docs + Prompt Version)"]
    K --> L
    
    L --> M[LLM Answer]
    
    M --> N[User]
    
    subgraph Retrieval [Retrieval Stage]
        B
        C
        D
        E
        F
        G
    end
    
    subgraph Context_Prompt [Context and Prompt Stage]
        H
        I
    end
    
    subgraph Generation [Generation Stage]
        J
        K
    end
    
    %% Force dark text on all nodes and subgraph titles for readability
    classDef default fill:#ffffff,stroke:#333,color:#000
    
    style A fill:#e1f5ff,stroke:#0288d1,color:#000
    style N fill:#e1f5ff,stroke:#0288d1,color:#000
    style M fill:#c8e6c9,stroke:#388e3c,color:#000
    style L fill:#c8e6c9,stroke:#388e3c,color:#000
    
    style B fill:#fff3e0,stroke:#f57c00,color:#000
    style C fill:#fff3e0,stroke:#f57c00,color:#000
    style D fill:#fff3e0,stroke:#f57c00,color:#000
    style E fill:#fff3e0,stroke:#f57c00,color:#000
    style F fill:#fff3e0,stroke:#f57c00,color:#000
    style G fill:#fff3e0,stroke:#f57c00,color:#000
    
    style H fill:#f3e5f5,stroke:#7b1fa2,color:#000
    style I fill:#f3e5f5,stroke:#7b1fa2,color:#000
    
    style J fill:#e8f5e9,stroke:#2e7d32,color:#000
    style K fill:#e8f5e9,stroke:#2e7d32,color:#000
    
    style Retrieval fill:#fff8e1,stroke:#f57c00,color:#000
    style Context_Prompt fill:#fce4ec,stroke:#7b1fa2,color:#000
    style Generation fill:#e0f2e9,stroke:#2e7d32,color:#000
```

**Attribution**: Claude Sonnet 4.6 for syntax help with the Mermaid code after prompting it with the sequential steps in the RAG pipeline.

---
## Project Structure

```text
├── app/
│   └── app.py                  # Streamlit application
├── data/
│   ├── raw/                    # Raw source data (gitignored)
│   └── processed/              # Parquet corpus + cached artifacts (gitignored)
├── feedback/
│   └── user_feedback.csv       # Persisted user relevance feedback
├── notebooks/
│   └── milestone1_exploration.ipynb   # EDA and preprocessing notebook
├── results/
│   └── milestone1_discussion.md       # Qualitative evaluation write-up
├── src/
│   ├── bm25.py                 # BM25 index construction and search
│   ├── semantic.py             # FAISS index construction and semantic search
│   ├── hybrid.py               # RRF-based hybrid search
│   └── utils.py                # Tokenization, I/O, and shared utilities
└── environment.yml             # Conda environment specification
```

---

## Dataset

This project uses the `meta_Electronics.jsonl` and `Electronics.jsonl` datasets from [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/), which are products from the **Electronics** category.

**Raw dataset sizes:**

- `Electronics.jsonl`: **~18.3 million** customer reviews
- `meta_Electronics.jsonl`: **~1.6 million** unique products

**Subset used:** Because embedding and indexing 1.6 M products is computationally prohibitive, the preprocessing notebook samples **200,000 products** controlled by the constant `MAX_PRODUCTS = 200_000` in `notebooks/milestone1_exploration.ipynb`. Products are selected by taking the first 200 k distinct `parent_asin` values from the metadata file (in file order). Only reviews whose `parent_asin` appears in that 200 k set are retained. This gives a **200,000-row retrieval corpus**  with one document per product.

Each document in the retrieval corpus is created by combining product metadata and review text into a single `retrieval_text` field. The following metadata columns are stored alongside each document for display:

| Field | Description |
| --- | --- |
| `parent_asin` | Unique product identifier |
| `product_title` | Full product name |
| `description` | Product description |
| `main_category` | Top-level product category |
| `store` | Brand/store name |
| `price` | Listed price |
| `average_rating` | Mean star rating |
| `rating_number` | Total number of ratings |
| `review_count` | Number of text reviews |
| `features` | Bullet-point product features |
| `categories` | Category hierarchy |
| `all_review_titles` | Concatenated review titles |
| `review_text_200` | First 200 characters of a representative review |

---

## Data Processing

All preprocessing is handled in `notebooks/milestone1_exploration.ipynb`, which outputs a `retrieval_corpus.parquet` file to `data/processed/`.

**Why Polars lazy evaluation?** The raw review file is ~18.3 M rows. Loading it eagerly with pandas would cause out-of-memory errors on most laptops. Instead, the notebook uses `pl.scan_ndjson()` to build a lazy query plan and `sink_parquet()` to stream results directly to disk in chunks, so that only the smaller subset of rows needed at each step are actually read into in RAM. This makes the full pipeline runnable on a standard laptop without any cloud compute.

The key preprocessing steps are:

1. **Streaming reads**: both JSONL files are opened with `pl.scan_ndjson()` (lazily to prevent OOM risk)
2. **Sampling**: 200,000 unique `parent_asin` values are selected with `.unique().limit(200_000)`; both the metadata and reviews frames are then filtered to that set via a semi-join
3. **Review aggregation**: reviews are grouped by `parent_asin`; review titles are joined into `all_review_titles` and the first 200 characters of combined review text are stored in `review_text_200`
4. **Joining**: grouped reviews are left-joined onto product metadata so every product gets one corpus row even if it has no reviews
5. **`retrieval_text` construction**: a single searchable string is built by concatenating the product title, main category, store, features, description, average rating, categories, all review titles, and the review text snippet so it is searchable by end uses.
6. **Persisting final corpus**: the final frame is streamed to `data/processed/retrieval_corpus.parquet` via `sink_parquet()`

At search time, `utils.py` handles two tokenization paths:

- **Python tokenizer** (`tokenize()`): used for short user queries: lowercases, strips non-alphanumeric characters, and removes English stop words
- **Polars vectorized tokenizer** (`polars_tokenize_expr()`): used for bulk corpus tokenization at index build time using Polars expressions for faster tokenization speed

---

## Retrieval Workflows

### BM25 (Keyword Search)

Implemented in `src/bm25.py` using the `rank_bm25` library.

**Build path:** The corpus is loaded from Parquet in chunks, tokenized using the Polars vectorized expression, and used to construct a `BM25Okapi` index. Both the tokenized corpus and the BM25 index are persisted to disk as pickle files so they only need to be built once.

**Fast path:** On future runs, `load_or_build_search_artifacts()` checks for an existing BM25 index and metadata pickle, and if both exist, they are both loaded directly without re-reading the corpus at all, saving a lot of build time. 

**Search:** The user's query is tokenized with the Python tokenizer, scored against the BM25 index with `get_scores()`, and the top-k results are selected using `np.argpartition` (avoids a full sort for speed).

### Semantic Search (Vector Search)

Implemented in `src/semantic.py` using `sentence-transformers` and `FAISS`.

**Model:** `all-MiniLM-L6-v2`: a lightweight sentence embedding model that produces 384-dimensional normalized embeddings well-suited for cosine similarity search.

**Index:** FAISS `IndexHNSWFlat` with inner product metric (equivalent to cosine similarity on normalized embeddings). HNSW was chosen over `IndexFlatIP` for its approximate nearest neighbour approach, which gives substantially faster query times on large corpora with minimal recall loss.

**Build path:** The corpus is processed in chunks. Each chunk is embedded and saved to disk as a `.npy` file. Once all chunks are processed, the embeddings are added to the FAISS index and persisted as a `.index` file. Metadata rows are saved as a parallel pickle.

**Fast path:** If the FAISS index and metadata pickle both exist, they are loaded directly. If only the index is missing but the embedding chunk `.npy` files exist, the index is rebuilt from those chunks without re-embedding the entire corpus.

**Search:** The query is embedded with the same model (with `normalize_embeddings=True`), and `index.search()` returns the top-k nearest neighbours by cosine similarity.

**Why chunk and persist embedding partitions?** Embedding 200,000 documents takes several minutes and can exhaust RAM if done in one pass. The build path splits the corpus into fixed-size chunks, embeds each one, and immediately writes the result to a numbered `.npy` file (`embedding_chunks/chunk_*.npy`). This gives the following benefits: (1) **persistence resilience**: if the process is interrupted for some reason, already-written chunks are not re-computed on the next run; (2) **less memory per chunk**: only one chunk of embeddings lives in RAM at a time during build; (3) **fast index rebuilds**: if the FAISS `.index` file is deleted but the `.npy` chunks still exist, the index is recreated from the saved chunks in seconds without re-embedding anything.

### Hybrid Search (RRF Fusion)

Implemented in `src/hybrid.py` using Reciprocal Rank Fusion.

BM25 and semantic similarity scores live on entirely different numeric scales, so raw score fusion is not meaningful. Instead, hybrid search fuses the **rank positions** from each retrieval method using the RRF formula:

```text
hybrid_score(doc) = 1 / (k + rank_bm25(doc)) + 1 / (k + rank_semantic(doc))
```

Where `k = 60` is the standard RRF smoothing constant that dampens the influence of very high-ranked results and prevents any single method from dominating.

Each retrieval method is run with a `candidate_multiplier` (default 3×) so that more candidates are considered before fusing each retrieval result together. This makes sure that documents ranked highly by one method but outside the naive top-k of the other method are still captured.

Documents are then ranked by their combined hybrid score in descending order.

---

## Environment Setup

This project uses a `conda` environment. To recreate it:

First clone the repo: 

```bash
git clone https://github.com/UBC-MDS/DSCI-575-amazon-review-retrieval-jasjot-karan.git
cd DSCI-575-amazon-review-retrieval-jasjot-karan
```

Then activate the environment:

```bash
conda env create -f environment.yml
conda activate amazon-retrieval
```

### Ollama Setup (required for the RAG generation step)

The LLM generation step runs **phi4-mini locally** via [Ollama](https://ollama.com). Install it before launching the app:

#### macOS

```bash
brew install ollama
# or download the installer from https://ollama.com/download
```

#### Linux

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**Windows**: download the installer from [https://ollama.com/download](https://ollama.com/download).

Then start the Ollama server and pull the model:

```bash
ollama serve          # start the local server (leave this running)
ollama pull phi4-mini # download the model (~2 GB, one-time)
ollama list           # confirm phi4-mini appears in the list
```

### Environment variables

Copy `.env.example` to `.env` and fill in any values (`TAVILY_API_KEY`) your local setup requires. You'll need a `TAVILY_API_KEY`. You can create one by signing up at
[tavily.com](https://tavily.com/) and generating a key from the dashboard.

```bash
cp .env.example .env
```

Key dependencies:

| Package | Version | Purpose |
| --- | --- | --- |
| `python` | 3.12 | Core runtime |
| `polars` | 1.39.3 | Fast DataFrame processing for corpus loading |
| `faiss-cpu` | 1.8.* | Vector index for semantic search |
| `sentence-transformers` | 3.0.1 | Text embeddings |
| `rank-bm25` | 0.2.2 | BM25 implementation |
| `streamlit` | 1.36.0 | Web UI |
| `scikit-learn` | 1.5.0 | Stop words and utilities |
| `numpy` | 1.26.* | Numerical operations |

---

## Reproducing the Results

#### Option A: Download pre-built artifacts (recommended to reproduce the app ASAP)

Download `processed_data.zip` from the repo release: `Processed data artifacts v1.0.0 Latest` tag: `data-1.0.0`, unzip it, and move all the files inside into the `data/processed/` folder in the project directory. The zip extracts a folder where the files in the folder should be in `data/processed/` so the final layout looks like:

```text
data/processed/
├── retrieval_corpus.parquet
├── bm25_index.pkl
├── faiss_index.index
├── metadata_rows.pkl
└── embedding_chunks/
    └── chunk_*.npy
```

Once those files are in place you can skip the notebook and index-build steps entirely and run `streamlit run app/app.py` from the project directory to run the app on: `http://localhost:8501`

#### Option B: Build from raw data

### 1. Prepare the Data

Place the raw Amazon Electronics dataset files into `data/raw/`:

- `Electronics.jsonl` (~18.3 M reviews)
- `meta_Electronics.jsonl` (~1.6 M products)

Then run the preprocessing notebook to generate the retrieval corpus:

```bash
jupyter notebook notebooks/milestone1_exploration.ipynb
```

This streams both files using Polars lazy evaluation (`scan_ndjson`) and writes `data/processed/retrieval_corpus.parquet` containing 200,000 product documents.

### 2. Build the Search Indexes

The indexes are built automatically on first launch of the app (or by running the module scripts directly). To pre-build manually:

```bash
# Build BM25 index
python src/bm25.py

# Build FAISS semantic index
python src/semantic.py
```

Both scripts persist their artifacts to `data/processed/` so subsequent runs load instantly.

### 3. Run the App Locally

```bash
streamlit run app/app.py
```

The app will be available at `http://localhost:8501`. On first launch, it builds and caches all search artifacts automatically so that future loading of artifacts is almost instant.

---

## Usage Examples
 
Once the app is running, try the following example queries to see how each retrieval method performs:
 
| Query Type | Example | Expected Best Method |
|---|---|---|
| **Brand + model number** | `"Sony WH-1000XM5 noise cancelling"` | BM25 |
| **Technical spec** | `"USB 3.2 Gen 2 SSD 1TB"` | BM25 |
| **Intent / use-case** | `"headphones good for working from home"` | Semantic |
| **Problem-based** | `"camera for hiking and outdoor photography"` | Semantic |
| **Multi-constraint** | `"fast USB-C hub with HDMI for MacBook under $50"` | Hybrid |
 
---

## User Feedback & Evaluation
 
### Relevance Feedback
 
Every result card in the app includes a thumbs up (relevant) / thumbs down (not relevant) button. Clicking either appends a record to `feedback/user_feedback.csv` with the following fields:
 
- `query`, `search_type`, `rank`, `score`
- `parent_asin`, `product_title`, `store`, `price`, `average_rating`
This data can be used to compute precision@k and other IR metrics offline.
 
### Qualitative Evaluation
 
A 30-query evaluation set (10 easy, 10 medium, 10 difficult) and a detailed comparison of BM25 vs. semantic retrieval is documented in [results/milestone1_discussion.md](results/milestone1_discussion.md).
 
**Key findings:**
 
- **BM25** excels on technical/factoid queries with exact measurements, model numbers, or brand names
- **Semantic search** excels on intent-based natural language queries describing use cases or scenarios
- **Hybrid (RRF)** captures the strengths of both by rewarding documents ranked highly by either method
- Both methods degrade on complex multi-constraint queries, motivating future reranking work

## Authors

| Name | GitHub |
| --- | --- |
| Jasjot Parmar | [@jasjotp](https://github.com/jasjotp) |
| Karan Bains | [@kkaranbayns](https://github.com/karanbayns) |
