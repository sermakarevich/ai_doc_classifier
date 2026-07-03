from __future__ import annotations

import os
from pathlib import Path

import fitz

from doc_extractor.models import Document


def load_pdf(path: str) -> Document:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    doc = fitz.open(str(p))
    pages: list[str] = []
    for page in doc:
        text = page.get_text()
        pages.append(text.strip())
    doc.close()

    return Document(
        path=path,
        pages=pages,
        text="\n\n".join(pages),
        n_pages=len(pages),
    )
