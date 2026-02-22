from typing import List, Dict, Tuple, Optional
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from .vectorstore import VectorStore
from .prompts import DEFAULT_QA_PROMPT
from .providers import get_embedding_client, get_llm


def build_retriever_and_chain(
    persist_dir: Optional[str] = None,
    collection_name: str = "default",
    k: int = 4,
    llm_model: Optional[str] = None,
    prompt: Optional[PromptTemplate] = None,
) -> Tuple[object, object]:
    """Build and return a (retriever, qa_chain) tuple.

    Args:
        persist_dir: optional Chroma persist directory
        collection_name: collection within Chroma
        k: number of documents to retrieve
        llm_model: optional model name (defaults per provider: gpt-3.5-turbo / llama2)
        prompt: optional PromptTemplate to use for the QA chain

    Returns:
        (retriever, qa_chain)
    """
    emb = get_embedding_client()
    vs = VectorStore(embedding_client=emb, persist_dir=persist_dir)
    retriever = vs.get_retriever(collection_name=collection_name, k=k)

    llm = get_llm(model_name=llm_model, temperature=0.0)
    prompt_to_use = prompt or DEFAULT_QA_PROMPT

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt_to_use},
    )

    return retriever, qa_chain


def answer_query(qa_chain, query: str) -> Dict:
    """Run the QA chain and return the chain output (answer + sources).

    Returns the raw chain output which usually contains 'result'/'answer' and
    'source_documents'.
    """
    res = qa_chain.invoke({"query": query})

    # Normalize the answer text
    answer_text = None
    if isinstance(res, dict):
        answer_text = res.get("result") or res.get("answer") or res.get("output_text")
    if answer_text is None:
        # Fall back to string conversion
        answer_text = str(res)

    # Extract source documents if available
    sources = []
    source_docs = None
    if isinstance(res, dict):
        source_docs = res.get("source_documents") or res.get("source_documents")
    if source_docs is None and hasattr(res, "source_documents"):
        source_docs = getattr(res, "source_documents")

    if source_docs:
        for d in source_docs:
            meta = getattr(d, "metadata", {}) if hasattr(d, "metadata") else {}
            text = getattr(d, "page_content", str(d)) if hasattr(d, "page_content") else str(d)
            sources.append({"source": meta.get("source"), "chunk": meta.get("chunk"), "text": text, "metadata": meta})

    return {"answer": answer_text, "sources": sources, "raw": res}
