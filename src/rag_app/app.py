import warnings

# Suppress urllib3 LibreSSL/OpenSSL warning on macOS (must be before any import that loads urllib3)
warnings.filterwarnings("ignore", message=".*OpenSSL.*")
warnings.filterwarnings("ignore", message=".*LibreSSL.*")

import os
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from dotenv import load_dotenv

from .chain import build_retriever_and_chain, answer_query

load_dotenv()

app = FastAPI(title="Text RAG Service")

# Serve static assets and query UI
_static_dir = Path(__file__).parent / "static"
if _static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")


@app.get("/")
def read_root():
    return {"message": "RAG service alive"}


@app.get("/query-page")
def query_page():
    """Serve the HTML query UI."""
    path = Path(__file__).parent / "static" / "query.html"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Query page not found")
    return FileResponse(path)


@app.on_event("startup")
def startup_event():
    # Try to initialize the QA chain if environment is configured.
    persist_dir = os.getenv("CHROMA_PERSIST_DIR", None)
    try:
        _, qa_chain = build_retriever_and_chain(persist_dir=persist_dir)
        app.state.qa_chain = qa_chain
    except Exception:
        app.state.qa_chain = None


def _run_query(q: str):
    """Shared logic for query endpoint."""
    qa_chain = getattr(app.state, "qa_chain", None)
    if qa_chain is None:
        raise HTTPException(
            status_code=503,
            detail="QA chain not initialized. Build the index and set the provider env vars (see .env.example).",
        )
    return answer_query(qa_chain, q)


@app.get("/query")
@app.post("/query")
async def query(q: str):
    """Run the RAG QA chain against the indexed store.

    Supports both GET and POST. Ensure you have indexed documents (run the indexer)
    and set the required API keys for your chosen EMBEDDING_PROVIDER / LLM_PROVIDER.
    """
    return _run_query(q)
