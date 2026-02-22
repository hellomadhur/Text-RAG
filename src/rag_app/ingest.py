from pathlib import Path
from typing import List, Dict, Iterable


def _read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        texts = []
        for page in reader.pages:
            txt = page.extract_text()
            if txt:
                texts.append(txt)
        return "\n".join(texts)
    except Exception:
        # fallback to binary read if pypdf unavailable or parsing fails
        return path.read_text(errors="ignore")


def _read_docx(path: Path) -> str:
    try:
        import docx

        doc = docx.Document(str(path))
        return "\n".join(p.text for p in doc.paragraphs)
    except Exception:
        return path.read_text(errors="ignore")


def _read_html(path: Path) -> str:
    try:
        from bs4 import BeautifulSoup

        html = path.read_text(encoding="utf-8", errors="ignore")
        soup = BeautifulSoup(html, "lxml")
        return soup.get_text(separator="\n")
    except Exception:
        return path.read_text(errors="ignore")


def load_file(path: str) -> Dict:
    """Load a single file and return a document dict with 'text' and 'source'."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    suffix = p.suffix.lower()
    if suffix in {".txt", ".md", ".rst"}:
        text = _read_text_file(p)
    elif suffix in {".pdf"}:
        text = _read_pdf(p)
    elif suffix in {".docx"}:
        text = _read_docx(p)
    elif suffix in {".html", ".htm"}:
        text = _read_html(p)
    else:
        # generic fallback
        try:
            text = p.read_text(encoding="utf-8")
        except Exception:
            text = p.read_text(errors="ignore")
    return {"text": text, "source": str(p)}


def load_documents(path_or_dir: str, recursive: bool = True) -> List[Dict]:
    """Load documents from a path or directory.

    - If `path_or_dir` is a file, returns a single-element list.
    - If it's a directory, walks files recursively and loads supported types.
    """
    p = Path(path_or_dir)
    docs: List[Dict] = []
    if p.is_file():
        docs.append(load_file(str(p)))
        return docs

    # directory: iterate files
    for f in p.rglob("*") if recursive else p.iterdir():
        if f.is_dir():
            continue
        # skip binary / non-text files by extension
        skip_suffixes = {
            ".exe", ".bin", ".dll",
            ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".ico", ".tiff", ".tif",
            ".svg", ".heic", ".avif",
        }
        if f.suffix.lower() in skip_suffixes:
            continue
        try:
            docs.append(load_file(str(f)))
        except Exception:
            # ignore unreadable files
            continue
    return docs


def load_text_files(directory: str, extensions=None) -> List[Dict]:
    """Backward-compatible function: load only text-like files from a directory."""
    if extensions is None:
        extensions = [".txt", ".md"]
    docs = []
    for d in load_documents(directory):
        if Path(d.get("source", "")).suffix.lower() in extensions:
            docs.append(d)
    return docs
