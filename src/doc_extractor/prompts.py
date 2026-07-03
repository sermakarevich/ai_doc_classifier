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
