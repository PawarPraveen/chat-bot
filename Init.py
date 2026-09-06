"""AI-Powered HR Resume Assistant — modular package.

Modules:
    config         — all settings in one place
    security       — path-traversal-safe file resolution
    embeddings     — shared SentenceTransformer loading/encoding
    resume_index   — parsing, chunking, and semantic search over the resume
    tools          — LangChain tool definitions
    feedback       — feedback log and in-context correction lookup
    agent          — system prompt, LLM backend, and the tool-calling loop
"""