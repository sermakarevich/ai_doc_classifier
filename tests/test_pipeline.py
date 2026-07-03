from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from doc_extractor.cli import create_parser
from doc_extractor.loader import load_pdf
from doc_extractor.merge import merge_extractions
from doc_extractor.models import ExtractedField, ExtractionResult, FieldHit, FieldSpec, OutputSchema, SchemaExtraction
from doc_extractor.pipeline import ExtractionConfig, run_extraction

ROOT = "/Users/sergii/git/ai_doc_classifier"
PDF_PATH = f"{ROOT}/the-next-big-arenas-of-competition-executive-summary-final.pdf"
SCHEMA_PATH = f"{ROOT}/schema_sample.json"


# ---- FakeProvider ----

class FakeProvider:
    """Provider whose structured() always returns the same SchemaExtraction regardless of prompt."""

    def __init__(self, name: str) -> None:
        self.name = name

    @property
    def base_url(self) -> str:
        return "http://127.0.0.1:11435"

    async def structured(self, prompt: str, response_model: type) -> SchemaExtraction:
        # Build a fixed SchemaExtraction covering all fields from the schema
        all_fields = [
            FieldHit(name=f.name, value=f"{f.name}_value", grounding=f"description")
            for f in self._fields
        ]
        return SchemaExtraction(fields=all_fields)


# ---- test_pipeline: run_extraction with FakeProvider ----

def _load_schema() -> OutputSchema:
    schema_dict = json.loads(Path(SCHEMA_PATH).read_text())
    return OutputSchema.model_validate(schema_dict)


@pytest.mark.asyncio
async def test_run_extraction_total_calls_four():
    """Two fakes x 2 calls = 4 total fan_out calls."""
    schema = _load_schema()
    fake1 = FakeProvider("fake_a")
    fake1._fields = schema.fields
    fake2 = FakeProvider("fake_b")
    fake2._fields = schema.fields

    config = ExtractionConfig(calls_per_provider=2)
    result = await run_extraction(
        pdf_path=PDF_PATH,
        schema=schema,
        providers=[fake1, fake2],
        config=config,
    )

    assert result.total_calls == 4


@pytest.mark.asyncio
async def test_run_extraction_all_schema_fields_present():
    """Every field in the schema must appear in result.fields."""
    schema = _load_schema()
    fake1 = FakeProvider("fake_a")
    fake1._fields = schema.fields
    fake2 = FakeProvider("fake_b")
    fake2._fields = schema.fields

    config = ExtractionConfig(calls_per_provider=2)
    result = await run_extraction(
        pdf_path=PDF_PATH,
        schema=schema,
        providers=[fake1, fake2],
        config=config,
    )

    field_names = {f.name for f in result.fields}
    for spec in schema.fields:
        assert spec.name in field_names, f"Missing field: {spec.name}"


@pytest.mark.asyncio
async def test_run_extraction_extraction_result_schema_id():
    """Result schema_id must match schema."""
    schema = _load_schema()
    fake1 = FakeProvider("fake_a")
    fake1._fields = schema.fields
    fake2 = FakeProvider("fake_b")
    fake2._fields = schema.fields

    config = ExtractionConfig(calls_per_provider=2)
    result = await run_extraction(
        pdf_path=PDF_PATH,
        schema=schema,
        providers=[fake1, fake2],
        config=config,
    )

    assert result.schema_id == schema.schema_id


@pytest.mark.asyncio
async def test_run_extraction_result_document_is_pdf_path():
    result = _load_schema()
    schema = result
    fake1 = FakeProvider("fake_a")
    fake1._fields = schema.fields
    fake2 = FakeProvider("fake_b")
    fake2._fields = schema.fields

    config = ExtractionConfig(calls_per_provider=2)
    result = await run_extraction(
        pdf_path=PDF_PATH,
        schema=schema,
        providers=[fake1, fake2],
        config=config,
    )

    assert result.document == PDF_PATH


@pytest.mark.asyncio
async def test_run_extraction_fields_have_candidates():
    """Each ExtractedField should have at least one ValueGroup candidate."""
    schema = _load_schema()
    fake1 = FakeProvider("fake_a")
    fake1._fields = schema.fields
    fake2 = FakeProvider("fake_b")
    fake2._fields = schema.fields

    config = ExtractionConfig(calls_per_provider=2)
    result = await run_extraction(
        pdf_path=PDF_PATH,
        schema=schema,
        providers=[fake1, fake2],
        config=config,
    )

    for ef in result.fields:
        assert len(ef.candidates) >= 1


# ---- test_cli: parser + arg parsing ----

def test_parser_returns_argument_parser():
    parser = create_parser()
    assert parser is not None


def test_parser_extract_subcommand_exists():
    parser = create_parser()
    args = parser.parse_args(["extract", "test.pdf", "--schema", "schema_sample.json"])
    assert args.command == "extract"
    assert args.pdf == "test.pdf"
    assert args.schema == "schema_sample.json"
    assert args.calls_per_provider == 2
    assert args.output is None


def test_parser_extract_with_all_options():
    parser = create_parser()
    args = parser.parse_args([
        "extract",
        "x.pdf",
        "--schema", "s.json",
        "--calls-per-provider", "3",
        "--output", "out.json",
    ])
    assert args.pdf == "x.pdf"
    assert args.schema == "s.json"
    assert args.calls_per_provider == 3
    assert args.output == "out.json"


def test_parser_extract_requires_schema():
    parser = create_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["extract", "x.pdf"])
