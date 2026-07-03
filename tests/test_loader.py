import pytest

from doc_extractor.loader import load_pdf

ROOT = "/Users/sergii/git/ai_doc_classifier"
PDF_PATH = f"{ROOT}/the-next-big-arenas-of-competition-executive-summary-final.pdf"


def test_load_pdf_pages():
    doc = load_pdf(PDF_PATH)
    assert doc.n_pages == 17
    assert len(doc.pages) == 17


def test_load_pdf_text_not_empty():
    doc = load_pdf(PDF_PATH)
    assert doc.text  # non-empty


def test_load_pdf_contains_arenas():
    doc = load_pdf(PDF_PATH)
    assert "arenas" in doc.text.lower()


def test_load_pdf_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_pdf("/bogus/nonexistent.pdf")
