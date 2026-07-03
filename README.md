# ai_doc_classifier

Multi-call, multi-model forecast extraction from documents. Renders a PDF's pages to
PNG images and sends them to several local vision-capable Ollama models multiple times,
then merges the answers into a consensus list of sector forecasts, each with a
confidence score.

Each forecast has: `sector_name`, `revenue_now`, `revenue_forecast`, `year_now`,
`year_forecast`, `cagr`, `profit`.

## Requirements

- Python 3.11+ and [uv](https://docs.astral.sh/uv/)
- Running [Ollama](https://ollama.com/) server (default `http://127.0.0.1:11435`)
  with vision-capable models (e.g. qwen3.6, gemma4)
- [just](https://github.com/casey/just) — optional, for the shortcuts below

## Usage

```bash
uv sync
uv run python -m doc_extractor extract document.pdf --output result.json
uv run python -m doc_extractor extract document.pdf --max-pages 6
```

Or via just:

```bash
just run                      # bundled sample PDF
just run my.pdf "--max-pages 6"
just test
```

## Configuration

Providers come from the `DOC_EXTRACTOR_PROVIDERS` env var (JSON list); default is
`qwen3.6:latest` (see `DEFAULT_PROVIDERS` in `src/doc_extractor/constants.py`):

```bash
export DOC_EXTRACTOR_PROVIDERS='[{"model": "qwen3.6:latest"}, {"model": "gemma4:latest"}]'
```

Calls per provider: `--calls-per-provider N`. Timeout: `--timeout SECONDS`.

## How it works

1. `loader.py` — render PDF pages to PNG images (PyMuPDF)
2. `fanout.py` — N parallel calls per provider with per-call timeout
3. `merge.py` — group forecasts across calls by sector name (normalization + LLM
   semantic merge of name variants), majority-vote each field, score = share of calls
   that saw the sector
4. Result printed as JSON (`ForecastResult`)
