# src/chatbot.py
import openai
from config.settings import OPENAI_COMPLETIONS_DEPLOYMENT

# from src.cosmos_db import query_vector_search

from src.retriever import hybrid_retrieve

from src.embeddings import generate_embedding

# Global conversation history for the session.
conversation_history = []

def summarize_history(history):
    """
    Summarizes earlier parts of the conversation succinctly.
    """
    prompt = "Summarize the following conversation succinctly, capturing only key points:\n"
    for msg in history:
        prompt += f"{msg['role']}: {msg['content']}\n"
    try:
        response = openai.ChatCompletion.create(
            engine=OPENAI_COMPLETIONS_DEPLOYMENT,
            messages=[{"role": "system", "content": prompt}],
            temperature=0.1,
            max_tokens=400
        )
        summary = response["choices"][0]["message"]["content"].strip()
        print("Summary generated:", summary, flush=True)
        return summary
    except Exception as e:
        print("Error in summarize_history:", e, flush=True)
        return ""

def generate_response(user_query: str):
    global conversation_history
    print("generate_response called with:", user_query, flush=True)

    # Append current user query to conversation history.
    conversation_history.append({"role": "user", "content": user_query})

    # 1) Generate embedding for the **current** query
    try:
        query_embedding = generate_embedding(user_query)
        print("Embedding generated", flush=True)
    except Exception as e:
        error_msg = f"Error generating embedding: {e}"
        print(error_msg, flush=True)
        conversation_history.append({"role": "assistant", "content": error_msg})
        return error_msg

    # 2) Retrieve top‑k relevant chunks from Cosmos
    try:
        # docs = query_vector_search(query_embedding, top_k=10)
        
        # Hybrid BM25 + vector retrieval
        docs = hybrid_retrieve(user_query)
        
        print("Relevant docs:", docs, flush=True)
        
    except Exception as e:
        error_msg = f"Error querying Cosmos DB: {e}"
        print(error_msg, flush=True)
        conversation_history.append({"role": "assistant", "content": error_msg})
        return error_msg

    # 3) Build system prompt
    system_prompt = """
You are an AKU Employee Assistant designed to answer questions based solely on the content of the company’s policy documents.
Your role is to interpret the user’s question, locate the most relevant policy excerpt, and respond clearly and concisely.

Please follow these instructions when providing your response:

1. **Use the policy documents**: Always search the provided policy documents for the answer. Extract the most pertinent information—sections, clauses, dates—as needed to give a precise reply.
2. **Match intent, not just keywords**: If the user’s phrasing doesn’t exactly match the document, interpret the intent (e.g. “external suppliers” ↔ “third‑party vendors”) and find the relevant policy text.
3. **Leverage conversation history**: If the user asks follow‑up questions, refer to earlier messages to maintain context and coherence.
4. **Be concise and structured**:
   - If multiple steps or bullet points help clarity (e.g., “To request access…”), format your answer as a numbered list or bullets.
   - Otherwise, keep answers to 2–3 sentences.
5. **Ask for clarification when needed**: If the question is vague or missing critical details, prompt the user for more information before attempting to answer.
6. **Out‑of‑scope handling**: If no relevant information exists in the documents, respond:
   “I’m sorry, I couldn’t locate that information in the policy documents. Could you please clarify or contact the appropriate department?”
7. **No external knowledge**: Do not draw on any sources beyond the provided policy documents.
8. **Maintain a professional tone**: Always be polite, formal, and focused on policy—never make personal remarks or use informal language.
"""

    messages = [{"role": "system", "content": system_prompt}]

    # 4) Inject each retrieved document chunk (up to 1000 chars)
    if docs:
        for doc in docs:
            excerpt = doc.get("content", "")[:2000]
            doc_context = (
                f"Document: {doc.get('document_name', 'N/A')}, "
                f"Section: {doc.get('section', 'N/A')}.\n"
                f"Excerpt:\n{excerpt}"
            )
            messages.append({"role": "system", "content": doc_context})
    else:
        messages.append({
            "role": "system",
            "content": "No relevant documents found."
        })

    # 5) (Optional) include a one‐sentence summary of prior chat if available
    if len(conversation_history) > 2:
        summary = summarize_history(conversation_history[:-1])
        messages.append({
            "role": "system",
            "content": f"Conversation so far (summarized): {summary}"
        })

    # 6) Always append the user’s current question _last_
    messages.append({"role": "user", "content": user_query})

    # 7) Call the OpenAI Chat API
    try:
        response = openai.ChatCompletion.create(
            engine=OPENAI_COMPLETIONS_DEPLOYMENT,
            messages=messages,
            temperature=0.1,
            max_tokens=450
        )
        answer = response["choices"][0]["message"]["content"].strip()
        print("Answer received:", answer, flush=True)
        print("Using GPT model:", OPENAI_COMPLETIONS_DEPLOYMENT)
    except Exception as e:
        answer = f"Error in ChatCompletion: {e}"
        print(answer, flush=True)

    # 8) Save and return
    conversation_history.append({"role": "assistant", "content": answer})
    return {"response": answer, "results": []}
