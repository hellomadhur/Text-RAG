# Text-RAG

RAG (retrieval-augmented generation) service: index documents, then query them via CLI or HTTP API. Supports multiple embedding and LLM providers (OpenAI, HuggingFace, Ollama).

## Setup

From the repo root:

```bash
scripts/setup_env.sh
# or manually:
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and configure your providers.

### Provider options

| Use case   | Option        | Config in `.env` |
|-----------|----------------|-------------------|
| **Embeddings** | OpenAI        | `EMBEDDING_PROVIDER=openai`, `OPENAI_API_KEY=sk-...` |
|             | HuggingFace (local) | `EMBEDDING_PROVIDER=huggingface` (no key) |
|             | Ollama (local) | `EMBEDDING_PROVIDER=ollama`, run `ollama pull nomic-embed-text` |
| **LLM**   | OpenAI        | `LLM_PROVIDER=openai`, `OPENAI_API_KEY=sk-...` |
|             | Ollama (local) | `LLM_PROVIDER=ollama`, run e.g. `ollama pull llama3.2:1b`, set `OLLAMA_LLM_MODEL=llama3.2:1b` |

If `OPENAI_API_KEY` is not set, the app defaults to **HuggingFace** for embeddings and **Ollama** for the LLM. See `.env.example` for all options.

## Indexing

Index a directory of documents (required before querying):

```bash
export PYTHONPATH="$(pwd)/src"
python -m src.rag_app.indexer /path/to/your/documents
```

Optional: `--persist_dir ./mychroma` to set the Chroma DB path. Re-run the indexer if you change the embedding provider or the source documents.

## CLI

Query the index from the command line:

```bash
export PYTHONPATH="$(pwd)/src"
python -m src.rag_app.cli "What is in the docs?"
# interactive mode
python -m src.rag_app.cli
```

Optional: `--model <name>` to override the LLM model, `--persist-dir`, `--collection`.

## API

Run the dev server:

```bash
export PYTHONPATH="$(pwd)/src"
uvicorn rag_app.app:app --reload --host 127.0.0.1 --port 8000
```

- **GET /** — health check  
- **POST /query?q=...** — run the RAG query  
- **GET /docs** — Swagger UI  

Ensure the index has been built and provider env vars are set, or the service returns 503.

## Docker

Build and run (suitable for OpenAI or when passing env from the host):

```bash
docker build -t text-rag .
docker run -p 8000:8000 --env-file .env text-rag
```

Or pass variables explicitly:

```bash
docker run -p 8000:8000 \
  -e EMBEDDING_PROVIDER=openai \
  -e LLM_PROVIDER=openai \
  -e OPENAI_API_KEY="$OPENAI_API_KEY" \
  -e CHROMA_PERSIST_DIR=/app/data \
  -v "$(pwd)/.chromadb:/app/data" \
  text-rag
```

For **Ollama**, you typically run the app on the host (so it can reach `localhost:11434`). To use Ollama from Docker, run Ollama in another container or on the host and set `OLLAMA_HOST` (e.g. `http://host.docker.internal:11434`) when running the app container.

## CI

A basic GitHub Actions workflow in `.github/workflows/ci.yml` runs `pytest`.
