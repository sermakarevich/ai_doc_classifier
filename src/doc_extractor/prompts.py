from __future__ import annotations

from .models import OutputSchema


def extraction_prompt(doc_text: str, schema: OutputSchema) -> str:
    # Truncate document text to first 60,000 chars
    truncated = doc_text[:60_000]

    # Build field listing
    fields_list = "\n".join(
        f"- {f.name}: {f.description} ({f.type})" for f in schema.fields
    )

    body = "\n".join([
        "Extract fields from the document below. Value must be a string ('true'/'false' for boolean, digits for numbers, comma-separated for lists); use null when the field is absent. Include a short verbatim grounding quote from the document for each non-null value. Respond ONLY with JSON with shape {\"fields\": [{\"name\":..., \"value\":..., \"grounding\":...}]}.",
        "",
        "Document (truncated to 60000 chars):",
        truncated,
        "",
        "Fields to extract:",
        fields_list,
        "",
        "Respond with JSON only.",
    ])
    return body


def merge_prompt(field_name: str, field_description: str, variants: list[str]) -> str:
    # List variants as numbered raw strings
    variant_lines = "\n".join(f"{i+1}. \"{v}\"" for i, v in enumerate(variants))

    return "\n".join([
        f"Merge the following variants of the field '{field_name}' ({field_description}) into semantically equivalent groups.",
        "",
        "Variants:",
        variant_lines,
        "",
        "Instruct: group semantically equivalent variants (same fact, different wording/formatting, abbreviation vs full form). Every input variant appears in exactly one group. canonical_value = most complete/precise variant of the group. Respond ONLY with JSON {\"groups\": [{\"canonical_value\":..., \"variants\": [...]}}].",
    ])
