"""
AI-Powered HR Resume Assistant — LangChain agent + Unstructured parsing +
embedding-based retrieval.

Pipeline:
  1. Parse the uploaded resume with Unstructured (open-source, local, layout-
     aware) — detects real document structure (Title, NarrativeText,
     ListItem...) instead of matching text against a fixed header keyword
     list, so it doesn't silently miss non-standard section headers.
  2. Group elements into sections using detected Titles, then split into
     sentence-level chunks for precise retrieval.
  3. Embed every chunk ONCE at load time with a local SentenceTransformer
     model; cache the embeddings for the session.
  4. The LLM agent calls a retrieval tool that does cosine-similarity search
     over the cached embeddings to ground every answer in the real document.
  5. Feedback loop: recruiters mark answers correct/incorrect after each
     response. Incorrect answers are persisted to disk with corrections, and
     future semantically-similar queries get warned via an injected prompt
     hint — an LLM-native equivalent of the original TF-IDF chatbot's
     upvote/downvote retraining loop (in-context correction, not retraining).

This is RAG-*shaped* (parse -> chunk -> embed -> retrieve -> ground) but
intentionally lightweight: no persistent vector database, single document,
in-memory embeddings. Accurate term for it: "embedding-based retrieval" /
"lightweight RAG pattern" — not a full production RAG system.

Dependencies:
    pip install "unstructured[pdf,docx]" sentence-transformers numpy \
                langchain langchain-openai

    Unstructured's local PDF parsing also needs a couple of system packages
    for best results (table/layout detection):
        Debian/Ubuntu: sudo apt-get install poppler-utils libmagic-dev
        macOS:         brew install poppler libmagic
    It will still work without them for straightforward text-based PDFs,
    but falls back to a lower-fidelity parse.

Run:
    python main.py
"""

import json
import os
import re
import subprocess
import time
from pathlib import Path

import numpy as np
from unstructured.partition.auto import partition
from sentence_transformers import SentenceTransformer

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

# ---------------------------------------------------------------------------
# LLM backend — pick one:
#   "openai"  — cloud, paid (gpt-4o-mini: $0.15/$0.60 per 1M tokens), most
#               reliable tool-calling. Needs OPENAI_API_KEY.
#   "groq"    — cloud, genuinely free tier (no credit card), hosts open-weight
#               models (Llama 3.3, GPT-OSS, etc.) on fast custom hardware.
#               Rate-limited, not unlimited. Needs GROQ_API_KEY (free signup
#               at console.groq.com).
#   "ollama"  — fully local, free, offline, no API key at all. Needs Ollama
#               running locally with a tool-calling-capable model pulled.
# ---------------------------------------------------------------------------
BACKEND = os.environ.get("CHATBOT_BACKEND", "ollama")

OPENAI_MODEL_NAME = "gpt-4o-mini"
GROQ_MODEL_NAME = "llama-3.3-70b-versatile"   # confirmed tool-calling support
OLLAMA_MODEL_NAME = os.environ.get("OLLAMA_MODEL_NAME", "llama3.1:8b")
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"      # local sentence-transformer, runs offline
TOP_K = 3                                      # how many chunks to retrieve per query
MIN_SIMILARITY = 0.2                           # relevance floor to filter out noise

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}
UPLOAD_DIR = Path(__file__).resolve().parent / "uploads"
FEEDBACK_PATH = Path(__file__).resolve().parent / "feedback_log.json"

FEEDBACK_SIMILARITY_THRESHOLD = 0.55  # only surface corrections for genuinely similar past queries


# ---------------------------------------------------------------------------
# 1. Secure file resolution (path-traversal guarded, unchanged approach)
# ---------------------------------------------------------------------------

def _safe_resolve(file_path: str) -> Path:
    """Resolve a filename against UPLOAD_DIR only, rejecting anything that
    escapes it (blocks '../' traversal and absolute-path tricks)."""
    UPLOAD_DIR.mkdir(exist_ok=True)
    candidate = (UPLOAD_DIR / Path(file_path).name).resolve()
    if UPLOAD_DIR.resolve() not in candidate.parents and candidate != UPLOAD_DIR.resolve():
        raise ValueError("Invalid file path — outside the allowed upload directory.")
    if candidate.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {candidate.suffix}. Use .pdf, .docx, or .txt.")
    return candidate


def print_help() -> None:
    print("Commands: upload <filename>, help, quit")
    print(f"Place resumes in: {UPLOAD_DIR}")


def chunk_plain_text_resume(path: Path) -> list[dict]:
    """Read TXT resumes directly without Unstructured or libmagic."""
    text = path.read_text(encoding="utf-8", errors="ignore")
    if not text.strip():
        return []

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


# ---------------------------------------------------------------------------
# 2. Parse + section-aware chunking via Unstructured
# ---------------------------------------------------------------------------

def chunk_resume_with_unstructured(path: Path) -> list[dict]:
    """
    Parse the document with Unstructured's layout model, which classifies
    each element (Title, NarrativeText, ListItem, Table, ...) using the
    document's actual visual structure — not keyword matching. Elements are
    grouped into sections by the nearest preceding Title, then split into
    sentence-level chunks for precise embedding retrieval.
    """
    elements = partition(filename=str(path))

    chunks: list[dict] = []
    current_section = "general"

    for element in elements:
        text = (element.text or "").strip()
        if not text:
            continue

        category = element.category  # e.g. "Title", "NarrativeText", "ListItem"

        if category == "Title":
            current_section = text.lower()
            continue

        for sentence in re.split(r"(?<=[.!?])\s+", text):
            sentence = sentence.strip()
            if len(sentence) > 3:
                chunks.append({"section": current_section, "text": sentence})

    return chunks


# ---------------------------------------------------------------------------
# 3. Embedding index — computed ONCE per resume, cached in memory
# ---------------------------------------------------------------------------

_embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

_RESUME_CHUNKS: list[dict] = []              # [{"section": ..., "text": ...}, ...]
_CHUNK_EMBEDDINGS: np.ndarray | None = None  # shape: (num_chunks, dim), L2-normalized


def load_and_index_resume(file_path: str) -> int:
    """
    Load a resume file, parse + chunk it with Unstructured, and embed every
    chunk exactly once. This is the expensive step — it runs a single time
    per resume, not per query, so repeated Q&A afterward stays fast.
    """
    global _RESUME_CHUNKS, _CHUNK_EMBEDDINGS

    path = _safe_resolve(file_path)
    if not path.exists():
        raise FileNotFoundError(f"No such file: {path.name}")

    if path.suffix.lower() == ".txt":
        _RESUME_CHUNKS = chunk_plain_text_resume(path)
    else:
        _RESUME_CHUNKS = chunk_resume_with_unstructured(path)

    if not _RESUME_CHUNKS:
        _CHUNK_EMBEDDINGS = None
        return 0

    texts = [c["text"] for c in _RESUME_CHUNKS]
    _CHUNK_EMBEDDINGS = _embedder.encode(
        texts, convert_to_numpy=True, normalize_embeddings=True
    )  # normalized -> dot product == cosine similarity

    return len(_RESUME_CHUNKS)


# ---------------------------------------------------------------------------
# 4. Retrieval tool — semantic cosine similarity, not keyword overlap
# ---------------------------------------------------------------------------

@tool
def search_resume(query: str) -> str:
    """Semantically search the currently loaded resume for content relevant
    to the query (skills, projects, education, or experience). Returns the
    most relevant sentence-level chunks from the actual uploaded document,
    ranked by embedding similarity rather than exact keyword overlap — so it
    can match questions phrased differently than the resume's wording."""
    if _CHUNK_EMBEDDINGS is None or not _RESUME_CHUNKS:
        return "No resume has been loaded yet."

    query_embedding = _embedder.encode(
        [query], convert_to_numpy=True, normalize_embeddings=True
    )[0]

    similarities = _CHUNK_EMBEDDINGS @ query_embedding
    top_indices = np.argsort(similarities)[::-1][:TOP_K]

    results = []
    for idx in top_indices:
        score = similarities[idx]
        if score < MIN_SIMILARITY:
            continue
        chunk = _RESUME_CHUNKS[idx]
        results.append(f"[{chunk['section'].title()}] {chunk['text']} (similarity: {score:.2f})")

    if not results:
        return "No sufficiently relevant content found in the resume for that query."
    return "\n".join(results)


TOOLS = [search_resume]
TOOL_REGISTRY = {t.name: t for t in TOOLS}


# ---------------------------------------------------------------------------
# 4b. Feedback loop — LLM-native equivalent of the original TF-IDF chatbot's
#     self-improving feedback loop.
#
#     The original project retrained a classifier on upvoted/downvoted
#     examples. There's no equivalent "retrain the weights" step for an LLM
#     you don't own (or even a local one, cheaply) — so instead this logs
#     recruiter corrections permanently to disk, and on every new query,
#     retrieves any *semantically similar* past correction using the same
#     embedding-similarity approach as search_resume, then injects it into
#     the prompt as a hint. This is in-context learning from feedback, not
#     retraining — an honest, different mechanism for a similar goal.
# ---------------------------------------------------------------------------

_FEEDBACK_LOG: list[dict] = []                    # [{"query", "answer", "was_correct", "correction", "timestamp"}]
_FEEDBACK_EMBEDDINGS: np.ndarray | None = None     # embeddings of the QUERY field, only for incorrect entries


def _load_feedback_log() -> None:
    """Load prior feedback from disk, if any, and rebuild the embedding
    index for past-incorrect queries so lookups work immediately."""
    global _FEEDBACK_LOG, _FEEDBACK_EMBEDDINGS

    if FEEDBACK_PATH.exists():
        with open(FEEDBACK_PATH, "r", encoding="utf-8") as f:
            _FEEDBACK_LOG = json.load(f)
    else:
        _FEEDBACK_LOG = []

    _rebuild_feedback_index()


def _rebuild_feedback_index() -> None:
    """Recompute embeddings for every logged query that was marked
    incorrect (those are the ones worth warning future queries about)."""
    global _FEEDBACK_EMBEDDINGS

    incorrect_entries = [e for e in _FEEDBACK_LOG if not e["was_correct"]]
    if not incorrect_entries:
        _FEEDBACK_EMBEDDINGS = None
        return

    queries = [e["query"] for e in incorrect_entries]
    _FEEDBACK_EMBEDDINGS = _embedder.encode(
        queries, convert_to_numpy=True, normalize_embeddings=True
    )


def log_feedback(query: str, answer: str, was_correct: bool, correction: str | None = None) -> None:
    """
    Record a recruiter's feedback on an answer and persist it to disk.

    was_correct=False entries become searchable "past mistakes" that future,
    similar queries get warned about before the LLM answers again.
    """
    entry = {
        "query": query,
        "answer": answer,
        "was_correct": was_correct,
        "correction": correction,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    _FEEDBACK_LOG.append(entry)

    with open(FEEDBACK_PATH, "w", encoding="utf-8") as f:
        json.dump(_FEEDBACK_LOG, f, indent=2)

    _rebuild_feedback_index()  # keep the lookup index in sync immediately


def check_past_corrections(query: str) -> str | None:
    """
    Look up whether a semantically similar query was previously answered
    incorrectly. Returns a hint string to inject into the prompt, or None
    if nothing sufficiently similar was found.
    """
    if _FEEDBACK_EMBEDDINGS is None:
        return None

    incorrect_entries = [e for e in _FEEDBACK_LOG if not e["was_correct"]]
    query_embedding = _embedder.encode(
        [query], convert_to_numpy=True, normalize_embeddings=True
    )[0]

    similarities = _FEEDBACK_EMBEDDINGS @ query_embedding
    best_idx = int(np.argmax(similarities))
    best_score = similarities[best_idx]

    if best_score < FEEDBACK_SIMILARITY_THRESHOLD:
        return None

    past = incorrect_entries[best_idx]
    hint = (
        f"Note: a previous, similarly-phrased question (\"{past['query']}\") "
        f"was answered incorrectly before: \"{past['answer']}\"."
    )
    if past.get("correction"):
        hint += f" The correct answer was: \"{past['correction']}\"."
    return hint


# ---------------------------------------------------------------------------
# 5. System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """
You are an HR Assistant helping recruiters review a candidate's resume.

Rules:
- Always call search_resume to find relevant content before answering —
  never answer about the candidate's skills, projects, education, or
  experience from assumption.
- The retrieved chunks include similarity scores. Trust chunks with higher
  scores more; if the top result still seems weakly related, say the resume
  may not clearly cover that topic rather than overstating confidence.
- If search_resume returns nothing relevant, say clearly that the resume
  doesn't contain that information — do not guess or fill in gaps.
- Once you have enough information, answer directly. Do not call the tool
  again if you already have what you need.
""".strip()


# ---------------------------------------------------------------------------
# 6. Agent loop
# ---------------------------------------------------------------------------

def _init_llm():
    """
    Returns the base chat model, before tools are bound.

    Groq and Ollama both use their dedicated LangChain integration classes
    directly rather than init_chat_model(..., model_provider=...) — several
    community reports show bind_tools() raising NotImplementedError when
    routed through init_chat_model for less-common providers. Importing
    ChatGroq / ChatOllama directly is the more reliable path.
    """
    if BACKEND == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model=GROQ_MODEL_NAME, temperature=0.0)
    elif BACKEND == "ollama":
        try:
            result = subprocess.run(
                ["ollama", "show", OLLAMA_MODEL_NAME],
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError("Ollama is not running or is not installed. Start Ollama, then try again.") from exc
        if result.returncode != 0:
            raise RuntimeError(
                f"Ollama model '{OLLAMA_MODEL_NAME}' is not installed. "
                f"Run: ollama pull {OLLAMA_MODEL_NAME}"
            )
        from langchain_ollama import ChatOllama
        return ChatOllama(model=OLLAMA_MODEL_NAME, temperature=0.0)
    else:
        return init_chat_model(OPENAI_MODEL_NAME, model_provider="openai", temperature=0.0)


def run_hr_assistant(user_message: str, max_tool_calls: int = 6) -> str:
    llm = _init_llm()
    llm_with_tools = llm.bind_tools(TOOLS)

    messages = [SystemMessage(content=SYSTEM_PROMPT)]

    correction_hint = check_past_corrections(user_message)
    if correction_hint:
        messages.append(SystemMessage(content=correction_hint))

    messages.append(HumanMessage(content=user_message))

    tool_call_count = 0

    while True:
        if tool_call_count >= max_tool_calls:
            messages.append(HumanMessage(
                content="Tool call limit reached. Answer now using only what you've already found."
            ))
            final = llm.invoke(messages)
            return final.content

        response = llm_with_tools.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            return response.content

        for tool_call in response.tool_calls:
            tool_call_count += 1
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_id = tool_call["id"]

            tool_fn = TOOL_REGISTRY.get(tool_name)
            result = tool_fn.invoke(tool_args) if tool_fn else f"Unknown tool: {tool_name}"

            messages.append(ToolMessage(content=str(result), tool_call_id=tool_id))


# ---------------------------------------------------------------------------
# 7. Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        try:
            num_chunks = load_and_index_resume(sys.argv[1])
            print(f"Resume loaded. I indexed {num_chunks} section(s).")
        except (ValueError, FileNotFoundError) as exc:
            print(f"Error: {exc}")
            sys.exit(1)
    else:
        print("HR Resume Assistant. Type 'upload <filename>' to load a resume.")
        print_help()

    _load_feedback_log()
    print(f"Loaded {len(_FEEDBACK_LOG)} past feedback entr{'y' if len(_FEEDBACK_LOG) == 1 else 'ies'}.\n")

    while True:
        q = input("You: ").strip()
        if q.lower() in ("quit", "exit"):
            break
        if not q:
            continue

        command, _, argument = q.partition(" ")
        if command.lower() == "help":
            print_help()
            continue
        if command.lower() == "upload":
            try:
                num_chunks = load_and_index_resume(argument.strip())
                print(f"Resume loaded. I indexed {num_chunks} section(s).")
            except (ValueError, FileNotFoundError) as exc:
                print(f"Error: {exc}")
            continue

        try:
            answer = run_hr_assistant(q)
        except Exception as exc:
            print(f"Assistant: I could not answer that. {exc}")
            continue
        print("Assistant:", answer)

        # --- Feedback capture: the interactive equivalent of your original
        #     project's upvote/downvote buttons ---
        verdict = input("Was this correct? (y/n/skip): ").strip().lower()
        if verdict == "y":
            log_feedback(q, answer, was_correct=True)
        elif verdict == "n":
            correction = input("What's the correct answer? (optional, Enter to skip): ").strip()
            log_feedback(q, answer, was_correct=False, correction=correction or None)
            print("Logged — future similar questions will be flagged with this correction.")
        print()