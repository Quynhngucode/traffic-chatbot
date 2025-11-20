from dotenv import load_dotenv
import gradio as gr

from src.llm_backend import StructuredAnswer, graph

# Load env
load_dotenv()


def format_answer(payload):
    if isinstance(payload, StructuredAnswer):
        return f"{payload.answer}\n(Nguồn: {payload.source})"
    return str(payload)


# Hàm chat
def chat_fn(user_input, history):
    """
    user_input: câu hỏi người dùng
    history: danh sách history [(user_msg, bot_msg), ...]
    """
    cleaned_input = user_input.strip()
    if not cleaned_input:
        return history  # nếu trống thì không trả lời

    state = {"question": cleaned_input, "context": [], "answer": ""}
    result = graph.invoke(state)
    bot_message = format_answer(result.get("answer", ""))

    history = history or []
    history.append((cleaned_input, bot_message))

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
