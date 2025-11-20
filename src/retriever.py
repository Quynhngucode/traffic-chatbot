from dataclasses import dataclass
from typing import List

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from src.config import (
    BM25_TOP_K,
    DOMAIN_KEYWORDS,
    EMBEDDING_TOP_K,
    HYBRID_WEIGHT,
    OUT_OF_DOMAIN_KEYWORDS,
    OUT_OF_DOMAIN_RESPONSE,
    RETRIEVAL_K_MAX,
    RETRIEVAL_K_MIN,
)
from src.vector_store import load_documents, load_vector_store

#hybrid retriever (FAISS + BM25) kèm guardrail phát hiện câu hỏi ngoài lĩnh vực, tự động điều chỉnh 
class DomainError(Exception):
    """Raised when câu hỏi nằm ngoài phạm vi luật giao thông."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def has_domain_keyword(question: str) -> bool:
    normalized = _normalize(question)
    return any(keyword and keyword in normalized for keyword in DOMAIN_KEYWORDS)


def is_explicitly_out_of_domain(question: str) -> bool:
    normalized = _normalize(question)
    return any(keyword and keyword in normalized for keyword in OUT_OF_DOMAIN_KEYWORDS)


@dataclass
class GuardrailResult:
    ok: bool
    message: str = ""


def guardrail(question: str) -> GuardrailResult:
    if is_explicitly_out_of_domain(question):
        return GuardrailResult(ok=False, message=OUT_OF_DOMAIN_RESPONSE)

    if has_domain_keyword(question):
        return GuardrailResult(ok=True)

    # Không tìm thấy từ khóa rõ ràng => vẫn cho phép để tránh chặn nhầm
    return GuardrailResult(ok=True)


def _needs_exact_match(question: str) -> bool:
    question_lower = _normalize(question)
    return any(token in question_lower for token in ["điều", "khoản", "điểm", "mức phạt"])


class HybridRetriever:
    def __init__(self):
        vector_store = load_vector_store()
        if vector_store is None:
            raise FileNotFoundError("Vector store chưa được tạo. Hãy chạy src/vector_store.py trước.")

        documents = load_documents()
        if not documents:
            raise FileNotFoundError("Không tìm thấy corpus cho BM25. Hãy tái tạo vector store.")

        self.bm25 = BM25Retriever.from_documents(documents)
        self.bm25.k = BM25_TOP_K

        self.vector_retriever = vector_store.as_retriever(search_kwargs={"k": EMBEDDING_TOP_K})

    def _determine_k(self, question: str) -> int:
        k = RETRIEVAL_K_MAX if _needs_exact_match(question) else RETRIEVAL_K_MIN
        return max(1, k)

    @staticmethod
    def _make_key(doc: Document) -> tuple:
        return doc.page_content, tuple(sorted(doc.metadata.items()))

    def _merge_results(self, vector_docs: List[Document], bm25_docs: List[Document]) -> List[Document]:
        scored = {}

        for rank, doc in enumerate(vector_docs, start=1):
            key = self._make_key(doc)
            score = HYBRID_WEIGHT * (1 / rank)
            scored[key] = (max(scored.get(key, (0, None))[0], score), doc)

        for rank, doc in enumerate(bm25_docs, start=1):
            key = self._make_key(doc)
            score = (1 - HYBRID_WEIGHT) * (1 / rank)
            if key in scored:
                prev_score, prev_doc = scored[key]
                scored[key] = (prev_score + score, prev_doc)
            else:
                scored[key] = (score, doc)

        merged = sorted(scored.values(), key=lambda item: item[0], reverse=True)
        return [doc for _, doc in merged]

    def retrieve(self, question: str) -> List[Document]:
        k = self._determine_k(question)
        self.vector_retriever.search_kwargs["k"] = k
        self.bm25.k = k
        vector_docs = self.vector_retriever.invoke(question)
        bm25_docs = self.bm25.invoke(question)
        merged_docs = self._merge_results(vector_docs, bm25_docs)
        return merged_docs[:k]

