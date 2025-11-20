import os
from dotenv import load_dotenv
from typing_extensions import List, TypedDict
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import START, StateGraph
from pydantic import BaseModel, Field

from src.retriever import HybridRetriever, guardrail

# -----------------------------
# 1️⃣ Load biến môi trường
# -----------------------------
load_dotenv()  #đọc file .env

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.0-flash")

# 2. Khởi tạo LLM
llm = ChatGoogleGenerativeAI(
    model=LLM_MODEL,
    google_api_key=GOOGLE_API_KEY,
    temperature=0.0,
    convert_system_message_to_human=True
)

# 3. Prompt template
template = """
Bạn là trợ lý AI chuyên về Luật Giao thông Việt Nam.
Mục tiêu: trả lời trực tiếp, rõ ràng và có dẫn chiếu cụ thể.

YÊU CẦU TRÌNH BÀY
1. Câu mở đầu phải nêu rõ đầy đủ các nguồn: "Căn cứ ... Điều/Khoản ...", nếu có nhiều điều khoản thì liệt kê ngay trong cùng một câu (ví dụ: "Căn cứ Điều 46 và khoản 1, khoản 2 Điều 35...").
2. Nếu nội dung gồm nhiều yêu cầu/điều kiện/bước, hãy liệt kê bằng gạch đầu dòng với dấu "- ".
3. Mỗi gạch đầu dòng chỉ chứa một ý súc tích (bao gồm số liệu, mức phạt, điều kiện... nếu có).
4. Không chép nguyên văn dài dòng; chỉ giữ lại thông tin thiết yếu.
5. Nếu context không đủ thông tin, nói rõ phần còn thiếu thay vì suy đoán.
6. Kết thúc với dòng "(Nguồn: …)" liệt kê lại các điều/khoản đã sử dụng theo cùng thứ tự như câu mở đầu.

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
retriever = HybridRetriever() 

# trả lời lịch sự khi out-of-domain, và graph thừa hưởng prompt bullet-list đã siết chặt trước đó.
def retrieve(state: State, top_k: int = 3):
    question = state["question"]
    guardrail_result = guardrail(question)
    if not guardrail_result.ok:
        return {"context": [], "answer": guardrail_result.message}

    retrieved_docs = retriever.retrieve(question)
    return {"context": retrieved_docs}


def generate(state: State):
    if state.get("answer"):
        return {"answer": state["answer"]}

    # Nối nội dung của các chunk
    docs_content = "\n\n".join(f"{doc.page_content}" for doc in state["context"])

    # Tạo message từ template
    prompt = ChatPromptTemplate.from_template(template=template)
    messages = prompt.format_messages(question=state["question"], context=docs_content)

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
