# HR Resume Assistant

A local RAG-based resume assistant. It parses an uploaded resume, creates local
sentence embeddings, retrieves the most relevant content, and asks an LLM to
answer using only that content.

## Requirements

- Python 3.10+
- Ollama installed and running
- A tool-calling model, for example:

```powershell
ollama pull llama3.1
```

Install Python dependencies:

```powershell
python -m pip install -r requirements.txt
```

## Run

Put `.pdf`, `.docx`, or `.txt` resumes in the `uploads` folder, then start:

```powershell
python main.py
```

Use the CLI:

```text
upload resume.pdf
What are the candidate's technical skills?
Tell me about the most recent project.
quit
```

You can also load a resume directly:

```powershell
python main.py resume.pdf
```

Files are restricted to the `uploads` directory. The assistant uses semantic
embedding retrieval before calling Ollama and does not display retrieval tool
calls to the user.

## Configuration

Edit `BACKEND` and `OLLAMA_MODEL_NAME` in `main.py` to change the provider or
model. The default is the free local Ollama backend with `llama3.1`.

## Project Files

- `main.py` - application, secure upload handling, indexing, retrieval, and chat
- `requirements.txt` - Python dependencies
- `uploads/` - local resumes to analyze
- `.gitignore` - excludes `.venv`, caches, and runtime state
