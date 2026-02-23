from rag_app.preprocess import normalize_text, preprocess, deduplicate_texts


def test_normalize_basic():
    s = "This   is\u201ctest\u201d"
    out = normalize_text(s)
    assert "test" in out


def test_preprocess_and_dedupe():
    a = "Hello world!\n\n"
    b = "Hello world!\n\n"
    texts, metas = deduplicate_texts([preprocess(a), preprocess(b)], metadatas=[{"id": 1}, {"id": 2}])
    assert len(texts) == 1
    assert metas[0]["id"] == 1
