import os
from dotenv import load_dotenv
from pathlib import Path
import pickle
import gradio as gr

from src.vector_store import load_vector_store, retrieve_chunks
from src.llm_backend import llm, template, StructuredAnswer

# Load env
load_dotenv()

# Load vector store
vector_store = load_vector_store()
if vector_store is None:
    raise ValueError("❌ Vector store chưa được tạo. Hãy chạy vector_store.py trước.")

# Hàm chat
def chat_fn(user_input, history):
    """
    user_input: câu hỏi người dùng
    history: danh sách history [(user_msg, bot_msg), ...]
    """
    if not user_input.strip():
        return history  # nếu trống thì không trả lời

    # 1. Lấy top chunk từ vector store
    top_chunks = retrieve_chunks(vector_store, user_input, k=5)

    # 2. Tạo context string cho LLM
    context_text = "\n\n".join(
        f"Tài liệu: {chunk.metadata.get('source', 'Không rõ')}\n{chunk.page_content}"
        for chunk in top_chunks
    )

    # 3. Tạo prompt
    from langchain_core.prompts import ChatPromptTemplate
    prompt = ChatPromptTemplate.from_template(template)
    messages = prompt.format_messages(question=user_input, context=context_text)

    # 4. LLM trả StructuredAnswer
    structured_llm = llm.with_structured_output(StructuredAnswer)
    structured_response: StructuredAnswer = structured_llm.invoke(messages)

    # 5. Convert thành string để Gradio hiểu
    bot_message = f"{structured_response.answer}\n(Nguồn: {structured_response.source})"

    # 6. Update history
    history = history or []
    history.append((user_input, bot_message))

    return history

# Launch Gradio chat interface
with gr.Blocks() as demo:
    gr.Markdown("## 🚦 Chatbot Hỏi Đáp Luật Giao Thông")
    chatbot = gr.Chatbot()
    msg = gr.Textbox(placeholder="Nhập câu hỏi...")
    clear = gr.Button("Clear")

    msg.submit(chat_fn, inputs=[msg, chatbot], outputs=[chatbot])
    clear.click(lambda: [], None, chatbot)

demo.launch(share=True)
