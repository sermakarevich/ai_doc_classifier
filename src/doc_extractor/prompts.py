from __future__ import annotations

from .models import OutputSchema

MAX_DOC_CHARS = 60_000


def extraction_prompt(doc_text: str, schema: OutputSchema) -> str:
    # Truncate document text to first 60,000 chars
    truncated = doc_text[:MAX_DOC_CHARS]

    # Build field listing
    fields_list = "\n".join(
        f"- {f.name}: {f.description} ({f.type})" for f in schema.fields
    )

    body = "\n".join([
        "Extract fields from the document below. Value must be a string ('true'/'false' for boolean, digits for numbers, comma-separated for lists); use null when the field is absent. Include a short verbatim grounding quote from the document for each non-null value. Respond ONLY with JSON with shape {\"fields\": [{\"name\":..., \"value\":..., \"grounding\":...}]}.",
        "",
        f"Document (truncated to {MAX_DOC_CHARS} chars):",
        truncated,
        "",
        "Fields to extract:",
        fields_list,
        "",
        "Respond with JSON only.",
    ])
    return body


def vision_extraction_prompt(schema: OutputSchema, n_pages: int) -> str:
    fields_list = "\n".join(
        f"- {f.name}: {f.description} ({f.type})" for f in schema.fields
    )

    return "\n".join([
        f"The attached {n_pages} images are consecutive pages of one document. Read them and extract the fields listed below.",
        "Value must be a string ('true'/'false' for boolean, digits for numbers, comma-separated for lists); use null when the field is absent. Include a short verbatim grounding quote from the document for each non-null value. Respond ONLY with JSON with shape {\"fields\": [{\"name\":..., \"value\":..., \"grounding\":...}]}.",
        "",
        "Fields to extract:",
        fields_list,
        "",
        "Respond with JSON only.",
    ])


def merge_prompt(field_name: str, field_description: str, variants: list[str]) -> str:
    # List variants as numbered raw strings
    variant_lines = "\n".join(f"{i+1}. \"{v}\"" for i, v in enumerate(variants))

    return "\n".join([
        f"Merge the following variants of the field '{field_name}' ({field_description}) into semantically equivalent groups.",
        "",
        "Variants:",
        variant_lines,
        "",
        "Instruct: group semantically equivalent variants (same fact, different wording/formatting, abbreviation vs full form). Every input variant appears in exactly one group. canonical_value = most complete/precise variant of the group. Respond ONLY with JSON {\"groups\": [{\"canonical_value\": ..., \"variants\": [...]}]}.",
    ])


FORECAST_FIELDS_INSTRUCTION = "\n".join([
    "Extract EVERY sector/market/arena forecast stated in the document. For each forecast return:",
    "- sector_name: name of the sector, market, or arena being forecast",
    "- revenue_now: current or base-year revenue / market size, as stated (e.g. '$4.9 trillion')",
    "- revenue_forecast: projected revenue / market size, as stated",
    "- year_now: base year of the current figure (digits)",
    "- year_forecast: target year of the projection (digits)",
    "- cagr: compound annual growth rate, as stated (e.g. '9-11%')",
    "- profit: profit or economic profit figure for the sector, if stated",
    "All values are strings; use null when a value is not stated. Do not invent numbers.",
    "Respond ONLY with JSON of shape {\"forecasts\": [{\"sector_name\": ..., \"revenue_now\": ..., \"revenue_forecast\": ..., \"year_now\": ..., \"year_forecast\": ..., \"cagr\": ..., \"profit\": ...}]}.",
])


def forecast_prompt(doc_text: str) -> str:
    truncated = doc_text[:MAX_DOC_CHARS]
    return "\n".join([
        FORECAST_FIELDS_INSTRUCTION,
        "",
        f"Document (truncated to {MAX_DOC_CHARS} chars):",
        truncated,
        "",
        "Respond with JSON only.",
    ])


def vision_forecast_prompt(n_pages: int) -> str:
    return "\n".join([
        f"The attached {n_pages} images are consecutive pages of one document.",
        FORECAST_FIELDS_INSTRUCTION,
        "",
        "Respond with JSON only.",
    ])
