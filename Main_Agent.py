"""
CLI entry point.

Usage:
    python -m hr_assistant.main [optional_initial_resume_path]

Once running, use:
    load <filename>     — load or switch to a resume (must be in uploads/)
    <any question>       — ask the agent about the currently loaded resume
    quit / exit           — end the session

Place resume files (PDF, DOCX, or TXT) in the uploads/ folder before
loading them by name.
"""

import sys

import Config as config
import Resume_index as resume_index
from Agent import run_hr_assistant
from Feedback import load_feedback_log, log_feedback, entry_count


def _try_load(path: str) -> None:
    try:
        num_chunks = resume_index.load_and_index_resume(path)
        num_sections = resume_index.section_count() if num_chunks else 0
        print(f"Loaded '{resume_index.loaded_filename()}' — "
              f"{num_chunks} chunk(s) across {num_sections} section(s).\n")
    except (ValueError, FileNotFoundError) as e:
        print(f"Could not load resume: {e}\n")


def main() -> None:
    print(f"Place PDF, DOCX, or TXT resumes in: {config.UPLOAD_DIR}\n")

    load_feedback_log()
    print(f"Loaded {entry_count()} past feedback entr{'y' if entry_count() == 1 else 'ies'}.\n")

    # Optional: load a resume immediately if a path was passed as an argument.
    if len(sys.argv) > 1:
        _try_load(sys.argv[1])

    print("Type 'load <filename>' to load a resume, ask a question once loaded, "
          "or 'quit' to exit.\n")

    while True:
        q = input("You: ").strip()
        if q.lower() in ("quit", "exit"):
            break
        if not q:
            continue

        if q.lower().startswith(("load ", "upload ")):
            prefix = "load " if q.lower().startswith("load ") else "upload "
            _try_load(q[len(prefix):].strip())
            continue

        answer = run_hr_assistant(q)
        print("Assistant:", answer)

        verdict = input("Was this correct? (y/n/skip): ").strip().lower()
        if verdict == "y":
            log_feedback(q, answer, was_correct=True)
        elif verdict == "n":
            correction = input("What's the correct answer? (optional, Enter to skip): ").strip()
            log_feedback(q, answer, was_correct=False, correction=correction or None)
            print("Logged — future similar questions will be flagged with this correction.")
        print()


if __name__ == "__main__":
    main()