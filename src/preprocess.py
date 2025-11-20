import re
from typing import Iterable, List
from langchain_core.documents import Document

def normalize_whitespace(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def normalize_text(text: str) -> str:
    """
    Chuẩn hóa nội dung nhằm giảm nhiễu cho cả BM25 và embedding.
    - Hạ chữ thường nhưng vẫn giữ nguyên ký tự đặc thù của tiếng Việt.
    - Loại bỏ khoảng trắng dư thừa.
    """
    lowered = text.lower()
    return normalize_whitespace(lowered)

def preprocess_documents(documents: Iterable[Document]) -> List[Document]:
    processed = []
    for doc in documents:
        new_doc = Document(
            page_content=normalize_text(doc.page_content),
            metadata=doc.metadata.copy(),
        )
        processed.append(new_doc)
    return processed

