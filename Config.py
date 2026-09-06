"""Central configuration — all tunable settings live here, nowhere else."""

from pathlib import Path

# ---------------------------------------------------------------------------
# LLM backend — pick one:
#   "openai"  — cloud, paid (gpt-4o-mini), most reliable tool-calling.
#   "groq"    — cloud, free tier, hosts open-weight models on fast hardware.
#   "ollama"  — fully local, free, offline. Least reliable tool-calling.
# ---------------------------------------------------------------------------
BACKEND = "ollama"   # "openai" | "groq" | "ollama"

OPENAI_MODEL_NAME = "gpt-4o-mini"
GROQ_MODEL_NAME = "llama-3.3-70b-versatile"
OLLAMA_MODEL_NAME = "llama3.1:8b"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 3                       # chunks retrieved per resume query
MIN_SIMILARITY = 0.2            # relevance floor for resume retrieval
FEEDBACK_SIMILARITY_THRESHOLD = 0.55  # relevance floor for past-correction matches

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
FEEDBACK_PATH = BASE_DIR / "feedback_log.json"

MAX_TOOL_CALLS = 6