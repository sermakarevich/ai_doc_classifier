from __future__ import annotations

# ---- providers ----
PROVIDERS_ENV_VAR = "DOC_EXTRACTOR_PROVIDERS"
DEFAULT_BASE_URL = "http://127.0.0.1:11435"
DEFAULT_PROVIDERS: list[dict[str, str]] = [
    {"model": "qwen3.6:latest"},
    {"model": "gemma4:latest"},
]

# ---- LLM calls ----
TEMPERATURE = 0.2
MAX_ATTEMPTS = 2

# ---- extraction ----
MAX_DOC_CHARS = 60_000
DEFAULT_CALLS_PER_PROVIDER = 2
DEFAULT_TIMEOUT_S = 1800.0
DEFAULT_MAX_PAGES = 12
DEFAULT_ZOOM = 2.0

# ---- forecast merging ----
FORECAST_VALUE_FIELDS = (
    "revenue_now", "revenue_forecast", "year_now", "year_forecast", "cagr", "profit"
)
