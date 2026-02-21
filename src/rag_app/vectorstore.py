from typing import List, Dict, Optional
import os
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


class VectorStore:
    """Wrapper around Chroma vector store for persistence and retrieval."""

    def __init__(self, embedding_client: Optional[Embeddings] = None, persist_dir: Optional[str] = None):
        self.embedding_client = embedding_client
        persist_dir = persist_dir or os.getenv("CHROMA_PERSIST_DIR", "./.chromadb")
        self._chroma = Chroma(persist_directory=persist_dir, embedding_function=embedding_client) if embedding_client else None

    def from_documents(self, docs: List[Dict], embeddings: Optional[List[List[float]]] = None, collection_name: str = "default"):
        texts = [d["text"] for d in docs]
        metadatas = [{k: v for k, v in d.items() if k != "text"} for d in docs]
        if self._chroma is None:
            raise ValueError("VectorStore requires an embedding client to create Chroma store")
        self._chroma.add_texts(texts=texts, metadatas=metadatas, collection_name=collection_name)

    def get_retriever(self, collection_name: str = "default", k: int = 4):
        """Return a LangChain retriever configured for the collection."""
        if self._chroma is None:
            raise ValueError("VectorStore requires an embedding client to query")
        # Chroma supports as_retriever which returns a Retriever object compatible
        # with LangChain chains.
        return self._chroma.as_retriever(search_kwargs={"k": k})

    def persist(self):
        """Persist the Chroma collection to disk (if available)."""
        if self._chroma is None:
            return
        try:
            self._chroma.persist()
        except Exception:
            # Chroma may persist automatically depending on the configuration
            pass

    def similarity_search(self, query: str, k: int = 4, collection_name: str = "default") -> List[Dict]:
        if self._chroma is None:
            raise ValueError("VectorStore requires an embedding client to query")
        docs = self._chroma.similarity_search(query, k=k, collection_name=collection_name)
        # Convert LangChain Document objects to plain dicts
        results = []
        for d in docs:
            meta = d.metadata if hasattr(d, "metadata") else {}
            results.append({"text": d.page_content if hasattr(d, "page_content") else str(d), **meta})
        return results

