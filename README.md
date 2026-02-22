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

### Using Ollama (local LLM and optional embeddings)

To use Ollama as your LLM (and optionally for embeddings), install Ollama, pull a model, and start the server.

**1. Download and install Ollama**

- **macOS / Windows:** Download the installer from [ollama.com](https://ollama.com) and run it. Ollama will be available in your applications and (on macOS) from the menu bar.
- **Linux:** Run the install script:
  ```bash
  curl -fsSL https://ollama.com/install.sh | sh
  ```

**2. Pull the LLM model (e.g. Llama 3.2 1B)**

```bash
ollama pull llama3.2:1b
```

This downloads ~1.3 GB. For embeddings with Ollama, also pull:

```bash
ollama pull nomic-embed-text
```

**3. Run Ollama**

- **macOS / Windows:** Ollama usually runs in the background after install. If it isn’t running, open the Ollama app from your applications or menu bar.
- **Linux / headless:** Start the server in the foreground (or run it as a service):
  ```bash
  ollama serve
  ```
  By default it listens on `http://127.0.0.1:11434`. Keep this terminal open while using the RAG app.

Then in `.env` set `LLM_PROVIDER=ollama` and `OLLAMA_LLM_MODEL=llama3.2:1b` (and optionally `EMBEDDING_PROVIDER=ollama` if you use `nomic-embed-text`).

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
- **GET /query-page** — HTML UI to query the RAG (ask questions and see answers + sources)  
- **GET /query?q=...** or **POST /query?q=...** — run the RAG query (both methods supported)  
- **GET /docs** — Swagger UI  

**Query from the browser:** Open **http://127.0.0.1:8000/query-page** to use the built-in query page. You can also call the API directly, e.g. `curl "http://127.0.0.1:8000/query?q=your%20question"`.

Ensure the index has been built and provider env vars are set, or the service returns 503.

## Docker

**Option 1: Pull the pre-built image from Docker Hub**

```bash
docker pull hellomadhur/text-rag:latest
```

Copy `.env.example` to `.env` and set your provider options (see **Provider options** above). Then run the container:

- **OpenAI (or other cloud providers):**
  ```bash
  docker run -p 8000:8000 --env-file .env hellomadhur/text-rag:latest
  ```

- **Using Ollama from Docker:** The app runs inside the container, so `localhost:11434` would point at the container, not your machine. Run Ollama on your host first, then start the container with `OLLAMA_HOST` set so it can reach the host’s Ollama. On Docker Desktop (Mac/Windows), use `host.docker.internal`:
  ```bash
  docker run -p 8000:8000 --env-file .env -e OLLAMA_HOST=http://host.docker.internal:11434 hellomadhur/text-rag:latest
  ```
  If you prefer not to use Docker with Ollama, run the app on the host instead (see **API** above) so it can use `localhost:11434` directly.

**Option 2: Build the image from source**

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

**Troubleshooting**

- **`ModuleNotFoundError: No module named 'langchain.chains'`** — Rebuild the image after pulling the latest code; requirements pin LangChain to 0.3.x so the import works.
- **`ConnectionError: Failed to connect to Ollama`** when querying — The container can’t reach Ollama. Use `-e OLLAMA_HOST=http://host.docker.internal:11434` (see above) and ensure Ollama is running on the host.

## CI

A basic GitHub Actions workflow in `.github/workflows/ci.yml` runs `pytest`.
