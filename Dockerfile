# Text-RAG API server. Set EMBEDDING_PROVIDER, LLM_PROVIDER, and API keys via env (e.g. --env-file .env).
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt
COPY . /app
ENV PYTHONPATH=/app/src
EXPOSE 8000
CMD ["uvicorn", "rag_app.app:app", "--host", "0.0.0.0", "--port", "8000"]
