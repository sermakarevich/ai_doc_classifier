from __future__ import annotations

from pathlib import Path

import fitz

from doc_extractor.models import Document


def render_pdf_pages(path: str, zoom: float = 2.0, max_pages: int | None = None) -> list[bytes]:
    """Render PDF pages to PNG bytes for vision models."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    doc = fitz.open(str(p))
    matrix = fitz.Matrix(zoom, zoom)
    images: list[bytes] = []
    for i, page in enumerate(doc):
        if max_pages is not None and i >= max_pages:
            break
        images.append(page.get_pixmap(matrix=matrix).tobytes("png"))
    doc.close()
    return images


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
