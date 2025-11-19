from pathlib import Path
import os
import pickle
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from src.config import PDF_FOLDER, VECTOR_FOLDER, VECTOR_FILE, CHUNK_SIZE, CHUNK_OVERLAP

# 1. Load PDF và tách chunk
def load_split_pdf_text(path_dir=PDF_FOLDER):
    pdf_dir = Path(path_dir)
    pdf_files = list(pdf_dir.glob("**/*.pdf"))
    all_chunks = []

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

        for idx, chunk in enumerate(chunks):
            chunk.metadata["source"] = str(file)
            chunk.metadata["chunk_id"] = idx
            all_chunks.append(chunk)

    return all_chunks


# 2. Tạo vector store
def create_vector_store(chunks, save_file=True):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    vector_store = InMemoryVectorStore(embeddings)
    vector_store.add_documents(chunks)

    if save_file:
        os.makedirs(VECTOR_FOLDER, exist_ok=True)
        path = os.path.join(VECTOR_FOLDER, VECTOR_FILE)
        with open(path, "wb") as f:
            pickle.dump(vector_store, f)
        print(f"✅ Vector store đã lưu vào {path}")

    return vector_store

def load_vector_store():
    path = os.path.join(VECTOR_FOLDER, VECTOR_FILE)
    if Path(path).exists():
        with open(path, "rb") as f:
            vector_store = pickle.load(f)
        print(f"✅ Vector store đã được load từ {path}")
        return vector_store
    return None

# 3. Lấy top k chunk theo query
def retrieve_chunks(vector_store, query, k=5):
    return vector_store.similarity_search(query, k=k)

if __name__ == "__main__":
    # Load & split PDF
    chunks = load_split_pdf_text()  # đã dùng default PDF_FOLDER từ config

    # Tạo vector store & lưu tự động
    vector_store = create_vector_store(chunks)  # save_file=True mặc định

    # 4️⃣ Thử query
    query = "có được chơi ma tuý lúc lái xe không?"
    top_chunks = retrieve_chunks(vector_store, query, k=5)

    for i, chunk in enumerate(top_chunks):
        print(f"Chunk {i+1}:")
        print(f"Source: {chunk.metadata['source']}, Chunk ID: {chunk.metadata['chunk_id']}")
        print(chunk.page_content)
        print("-" * 50)
