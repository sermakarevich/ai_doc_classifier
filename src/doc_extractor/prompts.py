from __future__ import annotations


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


def vision_forecast_prompt(n_pages: int) -> str:
    return "\n".join([
        f"The attached {n_pages} images are consecutive pages of one document.",
        FORECAST_FIELDS_INSTRUCTION,
        "",
        "Respond with JSON only.",
    ])
