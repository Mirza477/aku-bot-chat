# src/chatbot_utils.py
from .chatbot import generate_response

# Simple wrapper so that the Streamlit UI can import chatbot_query from here.
chatbot_query = generate_response
