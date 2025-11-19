import os

# Folder chứa PDF
PDF_FOLDER = os.getenv("PDF_FOLDER", "data")

# Folder lưu vector store
VECTOR_FOLDER = os.getenv("VECTOR_STORE_FOLDER", "embeddings")
VECTOR_FILE = os.getenv("VECTOR_STORE_FILE", "vector_store.pkl")

# Chunk settings
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))
