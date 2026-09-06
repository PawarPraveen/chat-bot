# HR Resume Assistant

A modular local RAG-based resume assistant. It parses an uploaded resume,
creates local sentence embeddings, retrieves relevant content, and asks an LLM
to answer using only the retrieved resume context.

## Requirements

- Python 3.10+
- Ollama installed and running
- A tool-calling model:

```powershell
ollama pull llama3.1:8b
```

Install Python dependencies:

```powershell
python -m pip install -r requirements.txt
```

## Run

Put `.pdf`, `.docx`, or `.txt` resumes in the `uploads` folder, then start:

```powershell
python Main_Agent.py
```

Use the CLI:

```text
upload resume.txt
What are the candidate's technical skills?
Tell me about the most recent project.
quit
```

You can also load a resume directly:

```powershell
python Main_Agent.py resume.pdf
```

Use `upload <filename>` or `load <filename>` while the program is running.
Files are restricted to the `uploads` directory. TXT files use a direct local
parser; PDF and DOCX files use Unstructured parsing. Semantic retrieval runs
before Ollama is called, and retrieval/tool details are hidden from users.

## Configuration

Edit `Config.py` to change the backend or model. The default is the free local
Ollama backend with `llama3.1:8b`.

## Project Files

- `Main_Agent.py` - CLI entry point and feedback prompts
- `Agent.py` - LLM tool-calling loop and grounding prompt
- `Resume_index.py` - resume parsing, chunking, and semantic retrieval
- `Embedding.py` - shared lazy-loaded embedding model
- `Tool.py` - LangChain retrieval tool
- `Feedback.py` - persistent feedback and correction lookup
- `Security.py` - upload path validation
- `Config.py` - centralized configuration
- `requirements.txt` - Python dependencies
- `uploads/` - local resumes to analyze
- `.gitignore` - excludes `.venv`, caches, and runtime state
