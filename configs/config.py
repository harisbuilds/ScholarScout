import os

MILVUS_COLLECTION    = "erasmus_university_programs"
PROFESSOR_COLLECTION = "professor_opportunities"
MILVUS_TOP_K         = 5

LLM_MODEL       = "gpt-4o"
LLM_FAST_MODEL  = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"
EMBED_DIM       = 1536
CHUNK_SIZE      = 4000
CHUNK_OVERLAP   = 200

MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
