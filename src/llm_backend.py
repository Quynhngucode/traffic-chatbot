import os
from dotenv import load_dotenv
from pathlib import Path
from typing_extensions import List, TypedDict
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import START, StateGraph
from pydantic import BaseModel, Field
import pickle

# -----------------------------
# 1️⃣ Load biến môi trường
# -----------------------------
load_dotenv()  # đọc file .env

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.0-flash")
VECTOR_FOLDER = os.getenv("VECTOR_STORE_FOLDER", "embeddings")
VECTOR_FILE = os.getenv("VECTOR_STORE_FILE", "vector_store.pkl")

# -----------------------------
# 2️⃣ Load vector store
# -----------------------------
vector_path = Path(VECTOR_FOLDER) / VECTOR_FILE
if vector_path.exists():
    with open(vector_path, "rb") as f:
        vector_store = pickle.load(f)
    print(f"✅ Vector store đã được load từ {vector_path}")
else:
    raise FileNotFoundError(f"Vector store không tìm thấy tại {vector_path}. Hãy tạo trước bằng vector_store.py")

# -----------------------------
# 3️⃣ Khởi tạo LLM
# -----------------------------
llm = ChatGoogleGenerativeAI(
    model=LLM_MODEL,
    google_api_key=GOOGLE_API_KEY,
    temperature=0.0,
    convert_system_message_to_human=True
)

# -----------------------------
# 4️⃣ Prompt template
# -----------------------------
template = """
Bạn là trợ lý AI chuyên về Luật Giao thông Việt Nam.
Hãy trả lời câu hỏi dựa trên nội dung được cung cấp trong context.

Ngữ cảnh:
{context}

Câu hỏi:
{question}

Câu trả lời:
"""

# -----------------------------
# 5️⃣ Định nghĩa dữ liệu trả về
# -----------------------------
class StructuredAnswer(BaseModel):
    answer: str = Field(description="Câu trả lời trực tiếp cho câu hỏi của người dùng")
    source: str = Field(description="Nguồn của câu trả lời")

# -----------------------------
# 6️⃣ Định nghĩa State
# -----------------------------
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

# -----------------------------
# 7️⃣ Các bước xử lý
# -----------------------------
def retrieve(state: State, top_k: int = 3):
    retrieved_docs = vector_store.similarity_search(state["question"], k=top_k)
    return {"context": retrieved_docs}

def generate(state: State):
    # Nối nội dung của các chunk
    docs_content = "\n\n".join(f"{doc.page_content}" for doc in state['context'])

    # Tạo message từ template
    prompt = ChatPromptTemplate.from_template(template=template)
    messages = prompt.format_messages(question=state['question'], context=docs_content)

    # Sử dụng structured output
    structured_llm = llm.with_structured_output(StructuredAnswer)
    structured_response = structured_llm.invoke(messages)

    return {"answer": structured_response}

# -----------------------------
# 8️⃣ Xây dựng Graph
# -----------------------------
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

# -----------------------------
# 9️⃣ Test nhanh (tuỳ chọn)
# -----------------------------
if __name__ == "__main__":
    question = "Không được vượt xe trong trường hợp nào?"
    state = {"question": question, "context": [], "answer": ""}
    
    # chạy graph
    result = graph.invoke(state)
    print(result["answer"])
