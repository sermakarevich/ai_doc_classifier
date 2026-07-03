from __future__ import annotations

from typing import Literal

from .models import ForecastExtraction, ForecastResult
from .prompts import forecast_prompt, vision_forecast_prompt
from .provider import OllamaProvider, load_providers
from .fanout import fan_out
from .loader import load_pdf, render_pdf_pages
from .merge import merge_forecasts

from pydantic import BaseModel

from .constants import (
    DEFAULT_CALLS_PER_PROVIDER,
    DEFAULT_MAX_PAGES,
    DEFAULT_TIMEOUT_S,
    DEFAULT_ZOOM,
)


class ExtractionConfig(BaseModel):
    calls_per_provider: int = DEFAULT_CALLS_PER_PROVIDER
    timeout_s: float = DEFAULT_TIMEOUT_S
    mode: Literal["text", "vision"] = "text"
    max_pages: int = DEFAULT_MAX_PAGES
    zoom: float = DEFAULT_ZOOM


async def run_extraction(
    pdf_path: str,
    providers: list[OllamaProvider] | None = None,
    config: ExtractionConfig | None = None,
) -> ForecastResult:
    providers = providers or load_providers()
    config = config or ExtractionConfig()

    if config.mode == "vision":
        images = render_pdf_pages(pdf_path, zoom=config.zoom, max_pages=config.max_pages)
        prompt = vision_forecast_prompt(n_pages=len(images))
        call = lambda p: p.structured(prompt, ForecastExtraction, images=images)
    else:
        doc = load_pdf(pdf_path)
        prompt = forecast_prompt(doc.text)
        call = lambda p: p.structured(prompt, ForecastExtraction)

    extractions = await fan_out(
        providers,
        call,
        calls_per_provider=config.calls_per_provider,
        timeout_s=config.timeout_s,
    )
    forecasts = await merge_forecasts(extractions, merge_provider=providers[0])
    return ForecastResult(
        document=pdf_path,
        total_calls=len(extractions),
        forecasts=forecasts,
    )
