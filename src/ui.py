# src/ui.py
import gradio as gr
from .chatbot import generate_response

def chatbot_ui():
    with gr.Blocks() as demo:
        # Maintain conversation state.
        state = gr.State([])
        chatbot = gr.Chatbot(label="Company Policy Chatbot")
        user_input = gr.Textbox(label="Enter your query:", lines=2)
        clear_btn = gr.Button("Clear Chat")



        def respond(query, history):
            answer = generate_response(query)
            history.append([query, answer])
            return "", history, history

        user_input.submit(
            respond,
            inputs=[user_input, state],
            outputs=[user_input, chatbot, state]
        )
        clear_btn.click(
            lambda: ("", [], []),
            inputs=None,
            outputs=[user_input, chatbot, state],
            queue=False
        )
    return demo
