import os
import pickle
from pathlib import Path
from typing import List, Optional

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DOCS_FILE,
    PDF_FOLDER,
    VECTOR_FILE,
    VECTOR_FOLDER,
    EMBEDDING_MODEL_NAME,
)
from src.preprocess import preprocess_documents


def _get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

#chuyển sang FAISS, chuẩn hóa chunk và lưu thêm corpus phục vụ BM25
# Load PDF và tách chunk
def load_split_pdf_text(path_dir: str = PDF_FOLDER) -> List[Document]:
    pdf_dir = Path(path_dir)
    pdf_files = sorted(pdf_dir.glob("**/*.pdf"))
    all_chunks: List[Document] = []

    for file in pdf_files:
        loader = PyPDFLoader(str(file))
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            separators=[
                r"\nChương [IVXLCDM]+\n",
                r"\nĐiều \d+[:\.]",
                r"\n\n",
                r"\n",
                r" ",
            ],
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            is_separator_regex=True,
        )

        chunks = text_splitter.split_documents(docs)
        chunks = preprocess_documents(chunks)

        for idx, chunk in enumerate(chunks):
            chunk.metadata["source"] = str(file)
            chunk.metadata["chunk_id"] = idx
            all_chunks.append(chunk)

    return all_chunks


def _persist_documents(docs: List[Document]) -> None:
    os.makedirs(VECTOR_FOLDER, exist_ok=True)
    docs_path = Path(VECTOR_FOLDER) / DOCS_FILE
    with open(docs_path, "wb") as f:
        pickle.dump(docs, f)
    print(f"✅ Đã lưu {len(docs)} chunk cho BM25 tại {docs_path}")


def load_documents() -> Optional[List[Document]]:
    docs_path = Path(VECTOR_FOLDER) / DOCS_FILE
    if docs_path.exists():
        with open(docs_path, "rb") as f:
            return pickle.load(f)
    return None


# Tạo vector store FAISS
def create_vector_store(chunks: List[Document], save_files: bool = True) -> FAISS:
    embeddings = _get_embeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)

    if save_files:
        os.makedirs(VECTOR_FOLDER, exist_ok=True)
        vector_store.save_local(VECTOR_FOLDER, index_name=VECTOR_FILE)
        _persist_documents(chunks)
        print(f"✅ Vector store đã lưu vào folder {VECTOR_FOLDER}/{VECTOR_FILE}")

    return vector_store


def load_vector_store() -> Optional[FAISS]:
    path = Path(VECTOR_FOLDER)
    faiss_file = path / f"{VECTOR_FILE}.faiss"
    index_file = path / f"{VECTOR_FILE}.pkl"

    if path.exists() and faiss_file.exists() and index_file.exists():
        embeddings = _get_embeddings()
        vector_store = FAISS.load_local(
            folder_path=str(path),
            embeddings=embeddings,
            index_name=VECTOR_FILE,
            allow_dangerous_deserialization=True,
        )
        print(f"✅ Vector store đã được load từ {faiss_file}")
        return vector_store
    return None


if __name__ == "__main__":
    chunks = load_split_pdf_text()
    create_vector_store(chunks)
