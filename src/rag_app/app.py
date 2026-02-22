import warnings

# Suppress urllib3 LibreSSL/OpenSSL warning on macOS (must be before any import that loads urllib3)
warnings.filterwarnings("ignore", message=".*OpenSSL.*")
warnings.filterwarnings("ignore", message=".*LibreSSL.*")

import os
from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv

from .chain import build_retriever_and_chain, answer_query

load_dotenv()

app = FastAPI(title="Text RAG Service")


@app.get("/")
def read_root():
    return {"message": "RAG service alive"}


@app.on_event("startup")
def startup_event():
    # Try to initialize the QA chain if environment is configured.
    persist_dir = os.getenv("CHROMA_PERSIST_DIR", None)
    try:
        _, qa_chain = build_retriever_and_chain(persist_dir=persist_dir)
        app.state.qa_chain = qa_chain
    except Exception:
        app.state.qa_chain = None


@app.post("/query")
async def query(q: str):
    """Run the RAG QA chain against the indexed store.

    Ensure you have indexed documents (run the indexer) and set the required API keys
    for your chosen EMBEDDING_PROVIDER / LLM_PROVIDER (see .env.example).
    """
    qa_chain = getattr(app.state, "qa_chain", None)
    if qa_chain is None:
        raise HTTPException(
            status_code=503,
            detail="QA chain not initialized. Build the index and set the provider env vars (see .env.example).",
        )
    res = answer_query(qa_chain, q)
    return res
