from __future__ import annotations

from typing import Literal

from .models import ExtractionResult, OutputSchema, SchemaExtraction
from .prompts import extraction_prompt, vision_extraction_prompt
from .provider import OllamaProvider, load_providers
from .fanout import fan_out
from .loader import load_pdf, render_pdf_pages
from .merge import merge_extractions

from pydantic import BaseModel


class ExtractionConfig(BaseModel):
    calls_per_provider: int = 2
    timeout_s: float = 240.0
    mode: Literal["text", "vision"] = "text"
    max_pages: int = 12
    zoom: float = 2.0


async def run_extraction(
    pdf_path: str,
    schema: OutputSchema,
    providers: list[OllamaProvider] | None = None,
    config: ExtractionConfig | None = None,
) -> ExtractionResult:
    providers = providers or load_providers()
    config = config or ExtractionConfig()

    if config.mode == "vision":
        images = render_pdf_pages(pdf_path, zoom=config.zoom, max_pages=config.max_pages)
        prompt = vision_extraction_prompt(schema, n_pages=len(images))
        call = lambda p: p.structured(prompt, SchemaExtraction, images=images)
    else:
        doc = load_pdf(pdf_path)
        prompt = extraction_prompt(doc.text, schema)
        call = lambda p: p.structured(prompt, SchemaExtraction)

    extractions = await fan_out(
        providers,
        call,
        calls_per_provider=config.calls_per_provider,
        timeout_s=config.timeout_s,
    )
    fields = await merge_extractions(schema, extractions, merge_provider=providers[0])
    return ExtractionResult(
        schema_id=schema.schema_id,
        document=pdf_path,
        total_calls=len(extractions),
        fields=fields,
    )
