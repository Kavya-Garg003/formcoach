"""
Minimal RAG retriever over the curated biomechanics/coaching corpus
(PRD 6.3: "~30-50 documents ... indexed in FAISS").

Uses a hand-rolled TF-IDF vectorizer instead of a neural embedding model so
the whole project runs fully offline with zero extra downloads or API keys.
This is a deliberate, documented MVP choice -- swap `_vectorize` for a call
to a real embedding model (Voyage AI, OpenAI, sentence-transformers) later
and nothing else in this file needs to change, since FAISS just consumes
vectors.
"""
from __future__ import annotations
import re
import math
import json
from pathlib import Path
from collections import Counter
import numpy as np
import faiss

CORPUS_DIR = Path(__file__).resolve().parent.parent.parent / "corpus"
INDEX_CACHE = Path(__file__).resolve().parent.parent.parent / "data" / "rag_index.json"

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class RagStore:
    def __init__(self):
        self.doc_ids: list[str] = []
        self.doc_titles: list[str] = []
        self.doc_texts: list[str] = []
        self.vocab: dict[str, int] = {}
        self.idf: np.ndarray | None = None
        self.index: faiss.Index | None = None
        self._build()

    def _load_corpus(self):
        for path in sorted(CORPUS_DIR.glob("*.md")):
            text = path.read_text(encoding="utf-8")
            lines = text.strip().splitlines()
            title = lines[0].lstrip("# ").strip() if lines else path.stem
            self.doc_ids.append(path.stem)
            self.doc_titles.append(title)
            self.doc_texts.append(text)

    def _fit_tfidf(self):
        tokenized_docs = [_tokenize(t) for t in self.doc_texts]
        df: Counter = Counter()
        for tokens in tokenized_docs:
            df.update(set(tokens))

        n_docs = len(tokenized_docs)
        self.vocab = {term: i for i, term in enumerate(sorted(df.keys()))}
        self.idf = np.zeros(len(self.vocab), dtype="float32")
        for term, i in self.vocab.items():
            self.idf[i] = math.log((1 + n_docs) / (1 + df[term])) + 1.0

        vectors = np.zeros((n_docs, len(self.vocab)), dtype="float32")
        for row, tokens in enumerate(tokenized_docs):
            tf = Counter(tokens)
            for term, count in tf.items():
                if term in self.vocab:
                    vectors[row, self.vocab[term]] = count
        vectors *= self.idf
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vectors /= norms
        return vectors

    def _build(self):
        self._load_corpus()
        if not self.doc_texts:
            self.index = None
            return
        vectors = self._fit_tfidf()
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)

    def _vectorize_query(self, query: str) -> np.ndarray:
        tokens = _tokenize(query)
        vec = np.zeros((1, len(self.vocab)), dtype="float32")
        tf = Counter(tokens)
        for term, count in tf.items():
            if term in self.vocab:
                vec[0, self.vocab[term]] = count
        vec *= self.idf
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def search(self, query: str, k: int = 3) -> list[dict]:
        if self.index is None or self.index.ntotal == 0:
            return []
        q = self._vectorize_query(query)
        scores, idxs = self.index.search(q, min(k, self.index.ntotal))
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            results.append(
                {
                    "doc_id": self.doc_ids[idx],
                    "title": self.doc_titles[idx],
                    "text": self.doc_texts[idx],
                    "score": float(score),
                }
            )
        return results


_store: RagStore | None = None


def get_store() -> RagStore:
    global _store
    if _store is None:
        _store = RagStore()
    return _store
