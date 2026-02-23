import logging
import warnings

# Suppress urllib3 LibreSSL/OpenSSL warning on macOS (must be before any import that loads urllib3)
warnings.filterwarnings("ignore", message=".*OpenSSL.*")
warnings.filterwarnings("ignore", message=".*LibreSSL.*")

from typing import List
from pathlib import Path
import os
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from .ingest import load_documents
from .vectorstore import VectorStore
from .preprocess import preprocess, deduplicate_texts

logger = logging.getLogger(__name__)


def index_directory(source_dir: str, persist_dir: str = None, chunk_size: int = 1000, chunk_overlap: int = 200):
    logger.info("Indexing directory: %s", source_dir)

    # Load raw documents
    docs = load_documents(source_dir)
    logger.info("Loaded %d document(s) from %s", len(docs), source_dir)
    for d in docs:
        source = d.get("source", "?")
        logger.debug("  - %s", Path(source).name if source else "?")

    # Preprocess text content
    logger.info("Preprocessing and deduplicating documents...")
    for d in docs:
        d["text"] = preprocess(d.get("text", ""))

    # Deduplicate whole documents before chunking
    texts = [d["text"] for d in docs]
    metas = [{k: v for k, v in d.items() if k != "text"} for d in docs]
    texts, metas = deduplicate_texts(texts, metas)
    if len(texts) < len(docs):
        logger.info("After deduplication: %d document(s)", len(texts))

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    split_texts = []
    metadatas = []
    for d_text, meta in zip(texts, metas):
        chunks = splitter.split_text(d_text)
        for i, c in enumerate(chunks):
            split_texts.append(c)
            m = dict(meta)
            m.update({"chunk": i})
            metadatas.append(m)

    # Deduplicate chunks to reduce near-duplicate segments
    split_texts, metadatas = deduplicate_texts(split_texts, metadatas)
    logger.info("Split into %d chunk(s) (chunk_size=%d, overlap=%d)", len(split_texts), chunk_size, chunk_overlap)

    # build/store in Chroma using embedding function (Chroma will call it)
    from .providers import get_embedding_client
    from .vectorstore import _clear_collection_if_exists

    persist_path = persist_dir or os.getenv("CHROMA_PERSIST_DIR", "./.chromadb")
    _clear_collection_if_exists(persist_path, collection_name="default")
    logger.info("Connecting to embedding provider and creating vector store...")
    emb_client = get_embedding_client()
    vs = VectorStore(embedding_client=emb_client, persist_dir=persist_dir)
    docs_to_add = [{"text": t, **m} for t, m in zip(split_texts, metadatas)]
    logger.info("Embedding and storing %d chunk(s) in Chroma...", len(docs_to_add))
    vs.from_documents(docs_to_add)
    vs.persist()
    logger.info("Index saved to %s. Done.", persist_dir or os.getenv("CHROMA_PERSIST_DIR", "./.chromadb"))

    return vs


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    load_dotenv()
    p = argparse.ArgumentParser()
    p.add_argument("source_dir", help="Directory to index")
    p.add_argument("--persist_dir", default=None)
    p.add_argument("-v", "--verbose", action="store_true", help="Show debug logs (e.g. per-file names)")
    args = p.parse_args()
    if args.verbose:
        logging.getLogger(__name__).setLevel(logging.DEBUG)
    index_directory(args.source_dir, persist_dir=args.persist_dir)
