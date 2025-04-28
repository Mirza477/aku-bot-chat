# src/embeddings.py
import logging
import openai
from config.settings import OPENAI_ENDPOINT, OPENAI_KEY, OPENAI_EMBEDDINGS_DEPLOYMENT
from tenacity import retry, wait_random_exponential, stop_after_attempt

openai.api_type = "azure"
openai.api_base = OPENAI_ENDPOINT
openai.api_version = "2023-05-15"
openai.api_key = OPENAI_KEY

@retry(wait=wait_random_exponential(min=2, max=120), stop=stop_after_attempt(6))
def generate_embedding(text: str):
    try:
        response = openai.Embedding.create(
            engine=OPENAI_EMBEDDINGS_DEPLOYMENT,
            input=text
        )
        embedding = response["data"][0]["embedding"]
        return embedding
    except Exception as e:
        logging.error("Error generating embedding: %s", e)
        raise
