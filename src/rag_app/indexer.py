from typing import List
from pathlib import Path
import os

from langchain_text_splitters import RecursiveCharacterTextSplitter
from .ingest import load_documents
from .vectorstore import VectorStore
from .preprocess import preprocess, deduplicate_texts


def index_directory(source_dir: str, persist_dir: str = None, chunk_size: int = 1000, chunk_overlap: int = 200):
    # Load raw documents
    docs = load_documents(source_dir)

    # Preprocess text content
    for d in docs:
        d["text"] = preprocess(d.get("text", ""))

    # Deduplicate whole documents before chunking
    texts = [d["text"] for d in docs]
    metas = [{k: v for k, v in d.items() if k != "text"} for d in docs]
    texts, metas = deduplicate_texts(texts, metas)

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

    # build/store in Chroma using embedding function (Chroma will call it)
    from langchain_openai import OpenAIEmbeddings
    emb_client = OpenAIEmbeddings()
    vs = VectorStore(embedding_client=emb_client, persist_dir=persist_dir)
    docs_to_add = [{"text": t, **m} for t, m in zip(split_texts, metadatas)]
    vs.from_documents(docs_to_add)
    vs.persist()

    return vs


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("source_dir", help="Directory to index")
    p.add_argument("--persist_dir", default=None)
    args = p.parse_args()
    index_directory(args.source_dir, persist_dir=args.persist_dir)
