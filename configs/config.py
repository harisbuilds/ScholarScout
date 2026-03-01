"""
config.py – Central configuration for the Scholarship RAG system.
"""

# ── Milvus connection ──────────────────────────────────────────────────────────
MILVUS_URI = "./milvus_scholarship.db"   # Milvus Lite (file-based, zero setup)
# For a running Milvus server change to:
# MILVUS_URI  = "http://localhost:19530"
# MILVUS_TOKEN = "root:Milvus"

COLLECTION_NAME = "scholarships"

# ── Embedding model ────────────────────────────────────────────────────────────
# Free, local, no API key required. ~90 MB download on first run.
DENSE_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DENSE_DIM        = 384          # dimension for all-MiniLM-L6-v2

# ── Index / search parameters ──────────────────────────────────────────────────
DENSE_INDEX_TYPE   = "FLAT"       # FLAT is exact; use HNSW for large scale
DENSE_METRIC_TYPE  = "COSINE"
SPARSE_INDEX_TYPE  = "SPARSE_INVERTED_INDEX"
SPARSE_METRIC_TYPE = "BM25"

# ── Retrieval defaults ─────────────────────────────────────────────────────────
TOP_K = 10                        # candidates fetched from Milvus before rerank
FINAL_K = 5                       # results returned to the user

# ── Text field used for BM25 & TF-IDF ─────────────────────────────────────────
# This field concatenates all textual scholarship metadata for full-text search.
SEARCHABLE_TEXT_FIELD = "searchable_text"
