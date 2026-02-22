"""
Embedding and LLM provider factory. Configure via environment:

  EMBEDDING_PROVIDER  one of: openai (default), huggingface, ollama
  LLM_PROVIDER        one of: openai (default), ollama

OpenAI requires OPENAI_API_KEY. HuggingFace runs locally (no key). Ollama requires
a local Ollama server (ollama run nomic-embed-text for embeddings, ollama run llama2 for LLM).
"""
from typing import Optional
import os
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel


def _has_openai_key() -> bool:
    return bool((os.getenv("OPENAI_API_KEY") or "").strip())


def get_embedding_provider() -> str:
    p = (os.getenv("EMBEDDING_PROVIDER") or "").strip().lower()
    if p:
        return p
    # Default to free local embeddings when no OpenAI key (avoids quota errors)
    return "huggingface" if not _has_openai_key() else "openai"


def get_llm_provider() -> str:
    p = (os.getenv("LLM_PROVIDER") or "").strip().lower()
    if p:
        return p
    # Default to local Ollama when no OpenAI key (avoids quota errors)
    return "ollama" if not _has_openai_key() else "openai"


def get_embedding_client() -> Embeddings:
    """Return an embeddings client based on EMBEDDING_PROVIDER."""
    provider = get_embedding_provider()
    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings()
    if provider == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings
        model = os.getenv("HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        return HuggingFaceEmbeddings(model_name=model)
    if provider == "ollama":
        from langchain_ollama import OllamaEmbeddings
        model = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
        return OllamaEmbeddings(model=model)
    raise ValueError(
        f"Unknown EMBEDDING_PROVIDER={provider}. Use one of: openai, huggingface, ollama"
    )


def get_llm(
    model_name: Optional[str] = None,
    temperature: float = 0.0,
) -> BaseChatModel:
    """Return a chat LLM based on LLM_PROVIDER."""
    provider = get_llm_provider()
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model_name or os.getenv("OPENAI_LLM_MODEL", "gpt-3.5-turbo"),
            temperature=temperature,
        )
    if provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=model_name or os.getenv("OLLAMA_LLM_MODEL", "tinyllama"),
            temperature=temperature,
        )
    raise ValueError(
        f"Unknown LLM_PROVIDER={provider}. Use one of: openai, ollama"
    )
