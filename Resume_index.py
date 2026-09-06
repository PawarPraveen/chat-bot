"""
Resume parsing, chunking, and the embedding index for retrieval.

Holds the currently loaded resume's state in module-level variables — this
is a single-user, single-session design. A multi-user deployment would key
this by session/candidate ID instead of using globals.
"""

import re
from pathlib import Path

import numpy as np
from unstructured.partition.auto import partition

import Config as config
import Security as security
from Embedding import embed

_RESUME_CHUNKS: list[dict] = []              # [{"section": ..., "text": ...}, ...]
_CHUNK_EMBEDDINGS: np.ndarray | None = None  # shape: (num_chunks, dim), L2-normalized
_LOADED_FILENAME: str | None = None


def is_resume_loaded() -> bool:
    return _CHUNK_EMBEDDINGS is not None and len(_RESUME_CHUNKS) > 0


def loaded_filename() -> str | None:
    return _LOADED_FILENAME


def section_count() -> int:
    return len({c["section"] for c in _RESUME_CHUNKS})


def _chunk_with_unstructured(path: Path) -> list[dict]:
    """
    Parse the document with Unstructured's layout classifier (Title,
    NarrativeText, ListItem, ...) — real document structure, not keyword
    matching. Groups elements into sections by the nearest preceding Title,
    then splits into sentence-level chunks for precise embedding retrieval.

    strategy="fast" uses pdfminer-based text extraction only — skips
    Unstructured's hi-res layout-detection model, which otherwise downloads
    and runs an object-detection model on CPU and can make a normal
    text-based resume take minutes / spike CPU and RAM. "fast" is correct
    for standard text-based resumes; only scanned/image-based PDFs would
    need "hi_res" (OCR), at much heavier compute cost.
    """
    elements = partition(filename=str(path), strategy="fast")

    chunks: list[dict] = []
    current_section = "general"

    for element in elements:
        text = (element.text or "").strip()
        if not text:
            continue

        if element.category == "Title":
            current_section = text.lower()
            continue

        for sentence in re.split(r"(?<=[.!?])\s+", text):
            sentence = sentence.strip()
            if len(sentence) > 3:
                chunks.append({"section": current_section, "text": sentence})

    return chunks


def _chunk_plain_text(path: Path) -> list[dict]:
    """Parse TXT resumes directly so Unstructured/libmagic is not needed."""
    text = path.read_text(encoding="utf-8", errors="ignore")
    chunks: list[dict] = []
    current_section = "general"
    section_names = {
        "summary", "objective", "profile", "experience", "education",
        "skills", "projects", "certifications", "contact", "achievements",
    }
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        normalized = re.sub(r"[^a-z ]", "", line.lower()).strip()
        if normalized in section_names:
            current_section = normalized
            continue
        for sentence in re.split(r"(?<=[.!?])\s+", line):
            sentence = sentence.strip()
            if len(sentence) > 3:
                chunks.append({"section": current_section, "text": sentence})
    return chunks


def load_and_index_resume(file_path: str) -> int:
    """
    Load a resume file, chunk it, and embed every chunk exactly once.
    This is the expensive step — it runs a single time per resume, not
    per query, so repeated Q&A afterward stays fast.
    """
    global _RESUME_CHUNKS, _CHUNK_EMBEDDINGS, _LOADED_FILENAME

    path = security.safe_resolve(file_path)
    if not path.exists():
        raise FileNotFoundError(
            f"No such file: {path.name} (looked in {config.UPLOAD_DIR})"
        )

    _RESUME_CHUNKS = (_chunk_plain_text(path)
                      if path.suffix.lower() == ".txt"
                      else _chunk_with_unstructured(path))

    if not _RESUME_CHUNKS:
        _CHUNK_EMBEDDINGS = None
        _LOADED_FILENAME = None
        return 0

    texts = [c["text"] for c in _RESUME_CHUNKS]
    _CHUNK_EMBEDDINGS = embed(texts)
    _LOADED_FILENAME = path.name

    return len(_RESUME_CHUNKS)


def search(query: str) -> str:
    """Semantic search over the currently loaded resume. Returns the most
    relevant sentence-level chunks ranked by cosine similarity."""
    if not is_resume_loaded():
        return "No resume has been loaded yet."

    query_embedding = embed([query])[0]
    similarities = _CHUNK_EMBEDDINGS @ query_embedding
    top_indices = np.argsort(similarities)[::-1][:config.TOP_K]

    results = []
    for idx in top_indices:
        score = similarities[idx]
        if score < config.MIN_SIMILARITY:
            continue
        chunk = _RESUME_CHUNKS[idx]
        results.append(f"[{chunk['section'].title()}] {chunk['text']} (similarity: {score:.2f})")

    if not results:
        return "No sufficiently relevant content found in the resume for that query."
    return "\n".join(results)