import os
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
PDF_PATH = ROOT / "the-next-big-arenas-of-competition-executive-summary-final.pdf"


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.skipif(
    os.environ.get("RUN_INTEGRATION") != "1",
    reason="RUN_INTEGRATION=1 required to run integration tests",
)
async def test_live_forecast_extraction_on_mckinsey_pdf():
    """End-to-end: load PDF, fan-out LLM calls, merge forecasts, assert results."""
    from doc_extractor.pipeline import run_extraction, ExtractionConfig
    from doc_extractor.provider import load_providers

    assert PDF_PATH.exists(), f"Test PDF not found at {PDF_PATH}"

    result = await run_extraction(
        str(PDF_PATH),
        providers=load_providers(),
        config=ExtractionConfig(calls_per_provider=2),
    )

    assert result.total_calls >= 2, f"Expected total_calls >= 2, got {result.total_calls}"
    assert result.forecasts, "No forecasts extracted"

    top = result.forecasts[0]
    assert top.sector_name
    assert top.score > 0
    # Report forecasts sectors to 2040; at least one forecast should carry
    # a target year or projected revenue.
    assert any(f.year_forecast or f.revenue_forecast for f in result.forecasts)
