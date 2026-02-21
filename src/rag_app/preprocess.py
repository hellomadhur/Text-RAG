import re
import unicodedata
import hashlib
from typing import List, Tuple, Dict, Iterable, Optional

try:
    from langdetect import detect, DetectorFactory
    DetectorFactory.seed = 0
    _HAS_LANGDETECT = True
except Exception:
    _HAS_LANGDETECT = False


_WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Normalize unicode and unify whitespace/quotes."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    # normalize common quote characters
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    # replace multiple whitespace with single space
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()


def clean_text(text: str) -> str:
    """Basic cleaning: remove control chars and long runs of punctuation."""
    if not text:
        return ""
    # remove non-printable control characters
    text = ''.join(ch for ch in text if ch.isprintable() or ch == '\n' or ch == '\t')
    # collapse repeated punctuation (e.g., "......")
    text = re.sub(r"([!?.,;:\-])\1{2,}", r"\1", text)
    return text


def detect_language(text: str) -> Optional[str]:
    if not _HAS_LANGDETECT or not text:
        return None
    try:
        return detect(text)
    except Exception:
        return None


def preprocess(text: str) -> str:
    """Apply normalization and cleaning to text."""
    return clean_text(normalize_text(text))


def _short_fingerprint(text: str, length: int = 200) -> str:
    """Compute a short fingerprint for deduplication based on hash of text prefix."""
    h = hashlib.sha256()
    sample = text[:length].encode('utf-8', errors='ignore')
    h.update(sample)
    return h.hexdigest()


def deduplicate_texts(texts: Iterable[str], metadatas: Optional[Iterable[Dict]] = None) -> Tuple[List[str], List[Dict]]:
    """Remove exact/near-duplicates from a list of texts.

    Returns filtered (texts, metadatas) preserving order.
    """
    seen = set()
    out_texts: List[str] = []
    out_meta: List[Dict] = []
    metas = list(metadatas) if metadatas is not None else [None] * len(list(texts))
    for i, t in enumerate(texts):
        if not t:
            continue
        fp = _short_fingerprint(t)
        if fp in seen:
            continue
        seen.add(fp)
        out_texts.append(t)
        meta = metas[i] if i < len(metas) else None
        out_meta.append(meta or {})
    return out_texts, out_meta
