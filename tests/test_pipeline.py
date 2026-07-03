from __future__ import annotations

import pathlib
import tempfile

import pytest

from doc_extractor.cli import create_parser
from doc_extractor.models import Forecast, ForecastExtraction, ForecastResult
from doc_extractor.pipeline import ExtractionConfig, run_extraction

ROOT = "/Users/sergii/git/ai_doc_classifier"
PDF_PATH = f"{ROOT}/the-next-big-arenas-of-competition-executive-summary-final.pdf"


# ---- FakeProvider ----

class FakeProvider:
    """Provider whose structured() always returns the same ForecastExtraction."""

    def __init__(self, name: str, forecasts: list[Forecast] | None = None) -> None:
        self.name = name
        self._forecasts = forecasts or [
            Forecast(
                sector_name="E-commerce",
                revenue_now="$4 trillion",
                revenue_forecast="$11 trillion",
                year_now="2022",
                year_forecast="2040",
                cagr="9%",
            )
        ]

    @property
    def base_url(self) -> str:
        return "http://127.0.0.1:11435"

    async def structured(self, prompt: str, response_model: type, images=None) -> ForecastExtraction:
        return ForecastExtraction(forecasts=self._forecasts)


# ---- run_extraction with FakeProvider ----

@pytest.mark.asyncio
async def test_run_extraction_total_calls_four():
    """Two fakes x 2 calls = 4 total fan_out calls."""
    config = ExtractionConfig(calls_per_provider=2)
    result = await run_extraction(
        pdf_path=PDF_PATH,
        providers=[FakeProvider("fake_a"), FakeProvider("fake_b")],
        config=config,
    )
    assert result.total_calls == 4


@pytest.mark.asyncio
async def test_run_extraction_merges_same_sector():
    """All calls return the same sector -> one merged forecast, full score."""
    config = ExtractionConfig(calls_per_provider=2)
    result = await run_extraction(
        pdf_path=PDF_PATH,
        providers=[FakeProvider("fake_a"), FakeProvider("fake_b")],
        config=config,
    )
    assert isinstance(result, ForecastResult)
    assert len(result.forecasts) == 1
    fc = result.forecasts[0]
    assert fc.sector_name == "E-commerce"
    assert fc.revenue_forecast == "$11 trillion"
    assert fc.count == 4
    assert fc.score == 1.0


@pytest.mark.asyncio
async def test_run_extraction_result_document_is_pdf_path():
    config = ExtractionConfig(calls_per_provider=1)
    result = await run_extraction(
        pdf_path=PDF_PATH,
        providers=[FakeProvider("fake_a")],
        config=config,
    )
    assert result.document == PDF_PATH


# ---- CLI parser ----

def test_parser_extract_defaults():
    parser = create_parser()
    args = parser.parse_args(["extract", "test.pdf"])
    assert args.command == "extract"
    assert args.pdf == "test.pdf"
    assert args.calls_per_provider == 2
    assert args.output is None
    assert args.mode == "text"
    assert args.max_pages == 12


def test_parser_extract_with_all_options():
    parser = create_parser()
    args = parser.parse_args([
        "extract",
        "x.pdf",
        "--calls-per-provider", "3",
        "--output", "out.json",
        "--mode", "vision",
        "--max-pages", "6",
    ])
    assert args.pdf == "x.pdf"
    assert args.calls_per_provider == 3
    assert args.output == "out.json"
    assert args.mode == "vision"
    assert args.max_pages == 6


# ---- zero-calls guard ----

def test_main_zero_calls_exits_1(monkeypatch):
    """When all provider calls fail, total_calls==0 and main() exits 1."""
    from doc_extractor.cli import main
    from doc_extractor.models import ForecastResult

    fake_result = ForecastResult(
        document="x.pdf",
        forecasts=[],
        total_calls=0,
        extraction_time_s=None,
        model_versions={},
        provider_stats={},
    )

    def fake_run(coro):
        return fake_result

    monkeypatch.setattr("asyncio.run", fake_run)

    # Create a temporary pdf so the exists check passes
    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    tmp.close()
    monkeypatch.setattr("sys.argv", ["doc_extractor", "extract", tmp.name])

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 1

    pathlib.Path(tmp.name).unlink(missing_ok=True)
