from typing import List
import os

from langchain_openai import OpenAIEmbeddings


# Instantiate a LangChain embeddings client. It will read OPENAI_API_KEY from
# the environment (or you may pass it explicitly when needed).
_emb = OpenAIEmbeddings()


def get_embeddings(texts: List[str]) -> List[List[float]]:
    """Return embeddings for a list of texts using OpenAIEmbeddings.

    Args:
        texts: list of strings to embed

    Returns:
        List of embedding vectors (lists of floats).
    """
    return _emb.embed_documents(texts)
