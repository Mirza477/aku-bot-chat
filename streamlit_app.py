import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import streamlit as st
from src.chatbot_utils import chatbot_query

import streamlit as st
import nltk

# ensure punkt is available
nltk.download('punkt', quiet=True)

from nltk.tokenize import word_tokenize

# Configure the Streamlit page
st.set_page_config(
    page_title="AKU Employee Assistance",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS to enhance the look and feel
st.markdown("""
    <style>
    /* Overall Container Styles */
    .stApp {
        background: linear-gradient(135deg, #ffffff, #f2f2f2 80%);
        max-width: 900px;
        margin: 0 auto;
        font-family: 'Segoe UI', sans-serif;
    }
    .stApp > div {
        padding-top: 0 !important;
    }

    /* Chat Container Styling */
    .chat-container {
        background-color: #ffffff;
        border-radius: 15px;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.1);
        padding: 25px;
        margin: 30px 0;
        min-height: 500px;
        display: flex;
        flex-direction: column;
    }

    /* Message Styling */
    .stChatMessage {
        max-width: 95%;
        margin-bottom: 15px;
        border-radius: 12px;
        padding: 15px 20px;
        transition: transform 0.2s ease;
    }
    .stChatMessage:hover {
        transform: scale(1.01);
    }
    /* Hide default headers */
    .stChatMessage div[data-testid="stChatMessageHeader"] {
        display: none;
    }
    
    /* User Message */
    .stChatMessage.user-message {
        background-color: #e0f7fa;
        align-self: flex-end;
    }

    /* Assistant Message */
    .stChatMessage.assistant-message {
        background-color: #fff3e0;
        border-left: 5px solid #ffcc80;
        align-self: flex-start;
    }

    /* Sources Styling */
    .sources-container {
        background-color: #f9f9f9;
        border-radius: 5px;
        padding: 8px;
        margin-top: 8px;
        font-size: 0.9em;
        border: 1px solid #e0e0e0;
    }
    .sources-container a {
        color: #ff8f00;
        text-decoration: none;
    }
    .sources-container a:hover {
        text-decoration: underline;
    }

    /* Chat Input Styling */
    .stTextInput > div > div > input {
        border-radius: 25px;
        padding: 12px 18px;
        border: 2px solid #ffcc80;
        font-size: 1rem;
        transition: border-color 0.2s ease;
    }
    .stTextInput > div > div > input:focus {
        border-color: #ffb300;
    }

    /* Footer Styling */
    .footer {
        text-align: center;
        color: #666;
        padding: 20px;
        font-size: 0.9em;
        background-color: #fafafa;
        margin-top: 20px;
        border-top: 1px solid #ddd;
    }

    /* Header Styling */
    .header {
        text-align: center;
        margin-top: 20px;
        margin-bottom: 10px;
        color: #424242;
    }
    .header h1 {
        font-size: 2.8rem;
        margin: 0;
    }
    .header p {
        font-size: 1.1rem;
        margin: 0;
        color: #757575;
    }

    /* Hide default Streamlit hamburger menu */
    .stActionButton {
        display: none !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Header Section
st.markdown("""
    <div class="header">
        <h1>AKU Employee Assistance</h1>
        <p>AI-Powered Employee Support</p>
    </div>
    """, unsafe_allow_html=True)

# Initialize session state for conversation history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm the AKU Employee Assistance chatbot. How can I help you today?"}
    ]

# Main chat container
container = st.container()

with container:
    # Display existing chat messages
    for message in st.session_state.messages:
        if message["role"] == "user":
            with st.chat_message("user", avatar="👤"):
                st.markdown(message["content"])
        else:
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(message["content"])
                if message.get("sources"):
                    with st.expander("Sources"):
                        st.markdown(
                            f'<div class="sources-container"><ul>{"".join(message["sources"])}</ul></div>',
                            unsafe_allow_html=True
                        )

# Chat input
user_query = st.chat_input("Ask a question about employee support...")

if user_query:
    # Add user query to the session state
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    # Display the user query
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_query)
    
    # Process and display the assistant response
    with st.spinner("Thinking..."):
        try:
            response = chatbot_query(user_query)
            sources = []
            for source in response["results"]:
                sources.append(
                    f"""<li><a href="data/sample_cvs/{source["doc"]}" target="_blank">{source["doc"]}</a> (section {source["section"]}, relevance: {source["score"]:.2f})</li>"""
                )
            cleaned_response = response['response'].replace('\033[92m', '').replace('\033[0m', '').strip()
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(cleaned_response)
                if sources:
                    with st.expander("Sources"):
                        st.markdown(
                            f'<div class="sources-container"><ul>{"".join(sources)}</ul></div>',
                            unsafe_allow_html=True
                        )
            st.session_state.messages.append({
                "role": "assistant",
                "content": cleaned_response,
                "sources": sources
            })
        except Exception as e:
            error_message = f"I'm sorry, but I encountered an error: {e}. Could you please try your question again?"
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(error_message)
            st.session_state.messages.append({
                "role": "assistant",
                "content": error_message
            })

# Footer Section
st.markdown("""
    <div class="footer">
        AI-Powered Employee Support | © Service Delivery
    </div>
    """, unsafe_allow_html=True)
