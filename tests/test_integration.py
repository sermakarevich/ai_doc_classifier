from __future__ import annotations

import json
import os
import pathlib

import pytest

from doc_extractor.models import OutputSchema
from doc_extractor.pipeline import run_extraction, ExtractionConfig
from doc_extractor.provider import load_providers

ROOT = pathlib.Path(__file__).resolve().parent.parent
PDF_PATH = ROOT / "the-next-big-arenas-of-competition-executive-summary-final.pdf"
SCHEMA_PATH = ROOT / "schema_sample.json"


@pytest.fixture(scope="module")
def schema() -> OutputSchema:
    with open(SCHEMA_PATH) as f:
        data = json.load(f)
    return OutputSchema.model_validate(data)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_live_extraction_on_mckinsey_pdf(schema: OutputSchema):
    """End-to-end: load PDF, fan-out LLM calls, merge, assert results."""
    if os.environ.get("RUN_INTEGRATION") != "1":
        pytest.skip("RUN_INTEGRATION=1 required to run integration tests")

    assert PDF_PATH.exists(), f"Test PDF not found at {PDF_PATH}"
    assert schema.schema_id == "mckinsey_report"
    assert len(schema.fields) == 7

    providers = load_providers()
    config = ExtractionConfig(calls_per_provider=2)

    result = await run_extraction(
        str(PDF_PATH),
        schema,
        providers=providers,
        config=config,
    )

    # total_calls must be at least 2 (one provider x 2 calls minimum)
    assert result.total_calls >= 2, f"Expected total_calls >= 2, got {result.total_calls}"

    # All 7 schema fields must be present in result
    result_field_names = {f.name for f in result.fields}
    schema_field_names = {f.name for f in schema.fields}
    assert result_field_names == schema_field_names, \
        f"Missing fields: {schema_field_names - result_field_names}"
    assert len(result.fields) == 7
    assert result.schema_id == "mckinsey_report"

    # Title field must have a value and score >= 0.5
    title_field = next((f for f in result.fields if f.name == "title"), None)
    assert title_field is not None, "title field not found in result"
    assert title_field.value is not None, "title value is None"
    assert title_field.score >= 0.5, f"title score {title_field.score} < 0.5"
