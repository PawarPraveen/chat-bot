"""Shared embedding model — lazy-loaded once, reused by resume search and
the feedback-correction lookup, so the model is only downloaded/loaded a
single time regardless of which feature uses it first."""

from sentence_transformers import SentenceTransformer

import Config as config

_embedder: SentenceTransformer | None = None


def get_embedder() -> SentenceTransformer:
    """Load the embedding model on first actual use, not at import time."""
    global _embedder
    if _embedder is None:
        print(f"Loading embedding model '{config.EMBEDDING_MODEL_NAME}' "
              f"(first run downloads weights from Hugging Face — can take a minute)...")
        _embedder = SentenceTransformer(config.EMBEDDING_MODEL_NAME, device="cpu")
        print("Embedding model ready.")
    return _embedder


def embed(texts: list[str]):
    """Encode a list of texts into L2-normalized vectors (dot product ==
    cosine similarity)."""
    return get_embedder().encode(texts, convert_to_numpy=True, normalize_embeddings=True)