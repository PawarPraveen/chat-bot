"""File-path safety — restricts all file access to UPLOAD_DIR."""

from pathlib import Path

import Config as config


def safe_resolve(file_path: str) -> Path:
    """
    Resolve a filename against UPLOAD_DIR only, rejecting anything that
    escapes it (blocks '../' traversal and absolute-path tricks) and
    anything with an unsupported extension.
    """
    config.UPLOAD_DIR.mkdir(exist_ok=True)
    candidate = (config.UPLOAD_DIR / Path(file_path).name).resolve()

    if config.UPLOAD_DIR.resolve() not in candidate.parents and candidate != config.UPLOAD_DIR.resolve():
        raise ValueError("Invalid file path — outside the allowed upload directory.")

    if candidate.suffix.lower() not in config.SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: {candidate.suffix}. "
            f"Use one of: {', '.join(sorted(config.SUPPORTED_EXTENSIONS))}"
        )

    return candidate