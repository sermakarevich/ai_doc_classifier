# ai_doc_classifier

Multi-call, multi-model structured document extraction. Sends a PDF to several local
Ollama models multiple times, then merges the answers per field into a consensus value
with a confidence score and grounding quotes.

## Requirements

- Python 3.11+ and [uv](https://docs.astral.sh/uv/)
- Running [Ollama](https://ollama.com/) server (default `http://127.0.0.1:11435`)
- [just](https://github.com/casey/just) — optional, for the shortcuts below

## Usage

```bash
uv sync
uv run python -m doc_extractor extract document.pdf --schema schema_sample.json --output result.json
```

Or via just:

```bash
just run                      # bundled sample PDF + schema_sample.json
just run my.pdf my_schema.json
just test
```

## Configuration

Providers come from the `DOC_EXTRACTOR_PROVIDERS` env var (JSON list); default is
`qwen3.5:27b` and `gemma4:latest`:

```bash
export DOC_EXTRACTOR_PROVIDERS='[{"model": "qwen3.5:27b"}, {"model": "gemma4:latest"}]'
```

Schemas are JSON files with `schema_id` and `fields` (`name`, `description`, `type`) —
see `schema_sample.json`.

## How it works

1. `loader.py` — extract PDF text (PyMuPDF)
2. `fanout.py` — N parallel calls per provider with per-call timeout
3. `merge.py` — per field: normalize + group values, LLM-merge semantic duplicates, pick winner by vote count, score = count / total_calls
4. Result printed as JSON (`ExtractionResult`)
