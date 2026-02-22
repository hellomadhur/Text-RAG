import warnings

# Suppress urllib3 LibreSSL/OpenSSL warning on macOS (must be before any import that loads urllib3)
warnings.filterwarnings("ignore", message=".*OpenSSL.*")
warnings.filterwarnings("ignore", message=".*LibreSSL.*")

import os
import argparse
from dotenv import load_dotenv

from .chain import build_retriever_and_chain, answer_query

load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="RAG CLI: query an indexed collection")
    parser.add_argument("query", nargs="?", help="Query text. If omitted, enters interactive mode")
    parser.add_argument("--persist-dir", default=os.getenv("CHROMA_PERSIST_DIR", None))
    parser.add_argument("--collection", default="default")
    parser.add_argument("--model", default=None, help="LLM model name (default: from .env per provider, e.g. OLLAMA_LLM_MODEL)")
    args = parser.parse_args()

    _, qa_chain = build_retriever_and_chain(
        persist_dir=args.persist_dir,
        collection_name=args.collection,
        llm_model=args.model,
    )

    if args.query:
        res = answer_query(qa_chain, args.query)
        print("Answer:\n", res.get("answer"))
        if res.get("sources"):
            print("\nSources:")
            for s in res["sources"]:
                print(f"- {s.get('source')} (chunk={s.get('chunk')})")
        return

    # Interactive REPL
    print("Entering interactive query mode. Type 'exit' to quit.")
    while True:
        q = input("query> ")
        if not q or q.strip().lower() in {"exit", "quit"}:
            break
        res = answer_query(qa_chain, q)
        print("Answer:\n", res.get("answer"))
        if res.get("sources"):
            print("\nSources:")
            for s in res["sources"]:
                print(f"- {s.get('source')} (chunk={s.get('chunk')})")


if __name__ == "__main__":
    main()
