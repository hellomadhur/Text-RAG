# Text-RAG

## Setup

Quick steps to create a virtual environment and install dependencies:

```bash
# from the repo root
scripts/setup_env.sh
# or manually:
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and set `OPENAI_API_KEY` and other settings.

Run the dev server:

```bash
export PYTHONPATH="$(pwd)/src"
uvicorn rag_app.app:app --reload --host 127.0.0.1 --port 8000

CLI
---
Query the index with the included CLI after indexing:

```bash
python -m src.rag_app.cli "What is in the docs?"
# or interactive mode
python -m src.rag_app.cli
```

Docker
---
Build and run the service in Docker:

```bash
docker build -t text-rag .
docker run -e OPENAI_API_KEY="$OPENAI_API_KEY" -p 8000:8000 text-rag
```

CI
--
A basic GitHub Actions workflow is included at `.github/workflows/ci.yml` which runs `pytest`.

```
