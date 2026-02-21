#!/usr/bin/env bash
# Run the FastAPI app locally
export PYTHONPATH="$(pwd)/src"
uvicorn rag_app.app:app --reload --host 127.0.0.1 --port 8000
