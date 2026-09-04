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

import re
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
BACKEND = "ollama"   # "openai" | "groq" | "ollama"

OPENAI_MODEL_NAME = "gpt-4o-mini"
GROQ_MODEL_NAME = "llama-3.3-70b-versatile"   # confirmed tool-calling support
OLLAMA_MODEL_NAME = "llama3.1"                # must be a tool-calling-capable pull

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"      # local sentence-transformer, runs offline
TOP_K = 3                                      # how many chunks to retrieve per query
MIN_SIMILARITY = 0.2                           # relevance floor to filter out noise

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}
UPLOAD_DIR = Path(__file__).resolve().parent / "uploads"


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
        raise ValueError(f"Unsupported file type: {candidate.suffix}. Use .pdf or .docx.")
    return candidate


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


def print_help() -> None:
    print("Commands: upload <filename>, help, quit")
    print(f"Place PDF, DOCX, or TXT resumes in: {UPLOAD_DIR}")


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
        from langchain_ollama import ChatOllama
        return ChatOllama(model=OLLAMA_MODEL_NAME, temperature=0.0)
    else:
        return init_chat_model(OPENAI_MODEL_NAME, model_provider="openai", temperature=0.0)


def run_hr_assistant(user_message: str, max_tool_calls: int = 6) -> str:
    llm = _init_llm()
    llm_with_tools = llm.bind_tools(TOOLS)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_message),
    ]

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
            print(f"Indexed resume into {num_chunks} chunk(s).\n")
        except (ValueError, FileNotFoundError) as exc:
            print(f"Error: {exc}")
            sys.exit(1)
    else:
        print("HR Resume Assistant. Type 'upload <filename>' to load a resume.")
        print_help()

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        command, _, argument = user_input.partition(" ")
        command = command.lower()

        if command in {"quit", "exit"}:
            print("Goodbye!")
            break
        if command == "help":
            print_help()
            continue
        if command == "upload":
            try:
                num_chunks = load_and_index_resume(argument.strip())
                if num_chunks:
                    print(f"Resume loaded. I indexed {num_chunks} sections. Ask your question.")
                else:
                    print("The resume did not contain readable text.")
            except (ValueError, FileNotFoundError) as exc:
                print(f"Error: {exc}")
            continue

        try:
            print(f"Assistant: {run_hr_assistant(user_input)}")
        except Exception as exc:
            print(f"Assistant: I could not answer that. {exc}")