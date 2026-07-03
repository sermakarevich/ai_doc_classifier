from __future__ import annotations

from .models import ExtractionResult, OutputSchema, SchemaExtraction
from .prompts import extraction_prompt
from .provider import OllamaProvider, load_providers
from .fanout import fan_out
from .loader import load_pdf
from .merge import merge_extractions

from pydantic import BaseModel


class ExtractionConfig(BaseModel):
    calls_per_provider: int = 2
    timeout_s: float = 240.0


async def run_extraction(
    pdf_path: str,
    schema: OutputSchema,
    providers: list[OllamaProvider] | None = None,
    config: ExtractionConfig | None = None,
) -> ExtractionResult:
    providers = providers or load_providers()
    config = config or ExtractionConfig()
    doc = load_pdf(pdf_path)
    prompt = extraction_prompt(doc.text, schema)
    extractions = await fan_out(
        providers,
        lambda p: p.structured(prompt, SchemaExtraction),
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
