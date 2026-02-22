from typing import List

from .providers import get_embedding_client


def get_embeddings(texts: List[str]) -> List[List[float]]:
    """Return embeddings for a list of texts using the configured provider.

    Set EMBEDDING_PROVIDER to openai, huggingface, or ollama (see .env.example).
    """
    return get_embedding_client().embed_documents(texts)
