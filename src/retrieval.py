"""Semantic retrieval of context notes for an anomalous route-week.

Retrieval is a *candidate generator*: it ranks notes by cosine similarity to a query
describing the anomaly. It never decides a verdict - see evidence_validation.py.

Backends
  * SentenceTransformerEmbedder (default): local all-MiniLM-L6-v2, CPU, no API.
  * TfidfEmbedder: lexical fallback / ablation baseline, used only when the model can't
    be loaded (or when asked for explicitly). The backend used is recorded in every run.

Ranking is made reproducible across machines by rounding scores to 6 decimals and
breaking ties on note_id.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


class EmbeddingModelUnavailable(RuntimeError):
    """The semantic model could not be loaded and fallback was not allowed."""


class SentenceTransformerEmbedder:
    backend = "sentence-transformers"

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer  # heavy import, keep lazy

        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device="cpu")
        self.tokenizer = self.model.tokenizer
        self.tokens_embedded = 0
        self.texts_embedded = 0

    def fit(self, corpus: list[str]) -> None:
        pass

    def encode(self, texts: list[str]) -> np.ndarray:
        self.texts_embedded += len(texts)
        self.tokens_embedded += sum(len(self.tokenizer(t)["input_ids"]) for t in texts)
        emb = self.model.encode(texts, batch_size=32, normalize_embeddings=True,
                                convert_to_numpy=True, show_progress_bar=False)
        return emb.astype(np.float64)


class TfidfEmbedder:
    backend = "tfidf"

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.model_name = "sklearn TfidfVectorizer(word 1-2 grams, sublinear_tf)"
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
        self.tokens_embedded = 0
        self.texts_embedded = 0

    def fit(self, corpus: list[str]) -> None:
        self.vectorizer.fit(corpus)

    def encode(self, texts: list[str]) -> np.ndarray:
        self.texts_embedded += len(texts)
        m = self.vectorizer.transform(texts).toarray()
        norms = np.linalg.norm(m, axis=1, keepdims=True)
        return np.divide(m, norms, out=np.zeros_like(m), where=norms > 0)


def get_embedder(kind: str, model_name: str, allow_fallback: bool = True):
    if kind == "tfidf":
        return TfidfEmbedder()
    try:
        return SentenceTransformerEmbedder(model_name)
    except Exception as exc:  # missing package, no cached weights and no network, ...
        if not allow_fallback:
            raise EmbeddingModelUnavailable(
                f"Could not load '{model_name}' ({type(exc).__name__}: {exc}). The first run needs internet "
                f"access to huggingface.co to download the model (~90 MB); later runs use the local cache."
            ) from exc
        log.warning("Sentence-transformer model unavailable (%s: %s); FALLING BACK to TF-IDF retrieval.",
                    type(exc).__name__, exc)
        return TfidfEmbedder()


def note_document(row) -> str:
    return f"Applies to: {row['applies_to']}. Date: {pd.Timestamp(row['date']):%d %B %Y}. {row['note']}"


def anomaly_query(route: str, week_of: pd.Timestamp, origin: str | None = None,
                  destination: str | None = None) -> str:
    if origin is None or destination is None:
        origin, _, destination = route.partition("-")
    return (f"Why did freight transport costs rise on the {route} route ({origin} to {destination}) "
            f"in the week of {pd.Timestamp(week_of):%d %B %Y}?")


@dataclass(frozen=True)
class Hit:
    note_id: str
    score: float
    rank: int


class NoteIndex:
    def __init__(self, notes: pd.DataFrame, embedder):
        self.embedder = embedder
        self.note_ids = notes["note_id"].tolist()
        docs = [note_document(r) for _, r in notes.iterrows()]
        if not docs:  # no notes: nothing can be retrieved, so every anomaly stays unexplained
            self.matrix = None
            return
        embedder.fit(docs)
        self.matrix = embedder.encode(docs)

    def search(self, query: str, k: int) -> list[Hit]:
        if self.matrix is None:
            return []
        q = self.embedder.encode([query])[0]
        scores = np.round(self.matrix @ q, 6)
        order = sorted(range(len(self.note_ids)), key=lambda i: (-scores[i], self.note_ids[i]))
        return [Hit(self.note_ids[i], float(scores[i]), rank) for rank, i in enumerate(order[:k], start=1)]

    def search_many(self, queries: list[str], k: int) -> list[list[Hit]]:
        """Batch version: one encode call for all queries."""
        if not queries:
            return []
        if self.matrix is None:
            return [[] for _ in queries]
        q = self.embedder.encode(queries)
        all_scores = np.round(q @ self.matrix.T, 6)
        out = []
        for scores in all_scores:
            order = sorted(range(len(self.note_ids)), key=lambda i: (-scores[i], self.note_ids[i]))
            out.append([Hit(self.note_ids[i], float(scores[i]), r) for r, i in enumerate(order[:k], start=1)])
        return out
