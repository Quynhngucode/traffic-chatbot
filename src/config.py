import os

# Đường dẫn dữ liệu & vector store
PDF_FOLDER = os.getenv("PDF_FOLDER", "data")
VECTOR_FOLDER = os.getenv("VECTOR_STORE_FOLDER", "embeddings")
VECTOR_FILE = os.getenv("VECTOR_STORE_FILE", "vector_store")
DOCS_FILE = os.getenv("DOCUMENT_STORE_FILE", "docs.pkl")

# Chunk & embedding settings
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
)

# Retrieval configs 
BM25_TOP_K = int(os.getenv("BM25_TOP_K", 6))
EMBEDDING_TOP_K = int(os.getenv("EMBEDDING_TOP_K", 6))
HYBRID_WEIGHT = float(os.getenv("HYBRID_WEIGHT", 0.5))  # 0=BM25, 1=vector
RETRIEVAL_K_MIN = int(os.getenv("RETRIEVAL_K_MIN", 4))
RETRIEVAL_K_MAX = int(os.getenv("RETRIEVAL_K_MAX", 10))

# Guardrail & domain configs
DOMAIN_KEYWORDS = [
    keyword.strip().lower()
    for keyword in os.getenv(
        "DOMAIN_KEYWORDS",
        "luật,giao thông,điều,khoản,xử phạt,bằng lái,sát hạch,phương tiện,xe,tốc độ,biển báo,đăng kiểm,đường bộ,ô tô,xe máy,giấy phép lái xe",
    ).split(",")
]

OUT_OF_DOMAIN_KEYWORDS = [
    keyword.strip().lower()
    for keyword in os.getenv(
        "OUT_OF_DOMAIN_KEYWORDS",
        "thời tiết,thể thao,bóng đá,game,chứng khoán,ẩm thực,du lịch,sức khỏe,tình yêu,phim ảnh",
    ).split(",")
]
OUT_OF_DOMAIN_RESPONSE = os.getenv(
    "OUT_OF_DOMAIN_RESPONSE",
    "Xin lỗi, tôi chỉ có thể tư vấn về Luật Giao thông Việt Nam. Bạn có thể đặt câu hỏi liên quan đến luật giao thông, biển báo, xử phạt, bằng lái…",
)
