# doc_extractor — simplified structured document extraction

Simplified clone of extractly's python extraction core. Scope: multi-call, multi-model,
semantic consensus merge with occurrence counts. No UI, no API service, no conflict
investigation.

## Architecture

```
PDF file
  → loader.py       pymupdf text extraction → Document(text, pages)
  → fanout.py       N providers × M calls_per_provider concurrent structured LLM calls
  → each call       prompts.py extraction prompt → provider.structured() → SchemaExtraction
  → merge.py        semantic merge: group equivalent values, count occurrences, score
  → ExtractionResult
```

All LLM access via local Ollama native API at `http://127.0.0.1:11435` using the
`format` parameter (JSON schema) for structured output.

## Package layout

```
pyproject.toml            # package name: doc-extractor, python >=3.11
src/doc_extractor/
    __init__.py
    models.py             # pydantic v2 models
    loader.py             # PDF → Document
    provider.py           # OllamaProvider + registry
    prompts.py            # prompt templates (plain python string functions)
    fanout.py             # concurrency primitive
    merge.py              # semantic consensus merge
    pipeline.py           # end-to-end orchestration
    cli.py                # python -m doc_extractor
tests/                    # pytest, unit tests use mocks — no network
schema_sample.json        # sample OutputSchema for the test PDF
```

Dependencies: `pydantic>=2`, `httpx`, `pymupdf`. Dev: `pytest`, `pytest-asyncio`.
Use `uv` conventions (PEP 621 pyproject). No langchain, no langgraph.

## models.py — exact contract (pydantic v2)

```python
class FieldSpec(BaseModel):
    name: str
    description: str
    type: Literal["string", "number", "integer", "boolean", "list"] = "string"

class OutputSchema(BaseModel):
    schema_id: str
    fields: list[FieldSpec]

class Document(BaseModel):
    path: str
    text: str                 # full concatenated text
    pages: list[str]          # per-page text
    n_pages: int

class FieldHit(BaseModel):    # one field value from one LLM call
    name: str
    value: str | None         # always string-serialized; None if not found
    grounding: str | None = None   # short quote from doc supporting the value

class SchemaExtraction(BaseModel):   # response model for one LLM extraction call
    fields: list[FieldHit]

class ValueGroup(BaseModel):  # aggregated candidate for one field
    canonical_value: str | None
    count: int                # how many calls produced a semantically equivalent value
    variants: list[str]       # raw distinct string variants merged into this group
    groundings: list[str]

class ExtractedField(BaseModel):
    name: str
    value: str | None         # canonical value of the winning (highest count) group
    count: int                # occurrences of winning group
    total_calls: int          # successful extraction calls
    score: float              # count / total_calls
    candidates: list[ValueGroup]   # all groups incl. winner, sorted by count desc

class ExtractionResult(BaseModel):
    schema_id: str
    document: str             # path
    total_calls: int
    fields: list[ExtractedField]

class MergeGroups(BaseModel):  # response model for the semantic-merge LLM call
    groups: list[MergeGroup]

class MergeGroup(BaseModel):
    canonical_value: str
    variants: list[str]        # exact raw variant strings assigned to this group
```

## provider.py — exact contract

```python
class ProviderConfig(BaseModel):
    name: str            # display name, defaults to model
    model: str           # e.g. "qwen3.5:27b"
    base_url: str = "http://127.0.0.1:11435"

class OllamaProvider:
    def __init__(self, config: ProviderConfig): ...
    @property
    def name(self) -> str: ...
    async def structured(self, prompt: str, response_model: type[T]) -> T:
        # POST {base_url}/api/chat with json:
        # {"model": model, "messages": [{"role":"user","content": prompt}],
        #  "stream": false, "format": response_model.model_json_schema(),
        #  "options": {"temperature": 0.2}}
        # parse resp["message"]["content"] as JSON → response_model.model_validate_json
        # on ValidationError/JSONDecodeError: retry once, then raise StructuredOutputError
        # use httpx.AsyncClient, timeout from call site

def load_providers(env_var: str = "DOC_EXTRACTOR_PROVIDERS") -> list[OllamaProvider]:
    # env var = JSON array of ProviderConfig dicts.
    # default if unset:
    # [{"model": "qwen3.5:27b"}, {"model": "gemma4:latest"}]

class StructuredOutputError(Exception): ...
```

## fanout.py — exact contract

```python
async def fan_out(
    providers: list[OllamaProvider],
    call: Callable[[OllamaProvider], Awaitable[T]],   # one attempt against one provider
    *,
    calls_per_provider: int = 2,
    timeout_s: float = 240.0,
) -> list[T]:
    # build len(providers)*calls_per_provider tasks: asyncio.wait_for(call(p), timeout_s)
    # asyncio.gather(..., return_exceptions=True)
    # log failures (logging.warning), return only successes
    # if ALL failed → raise AllCallsFailedError(with list of exceptions)
```

## prompts.py

```python
def extraction_prompt(doc_text: str, schema: OutputSchema) -> str
```
Instructs: extract each field (name + description listed), value as string, null if absent,
include short verbatim grounding quote. Respond with JSON matching provided schema.
Truncate doc_text to 60_000 chars.

```python
def merge_prompt(field_name: str, field_description: str, variants: list[str]) -> str
```
Instructs: group the raw variant strings into semantically equivalent groups
(e.g. "McKinsey Global Institute" == "MGI", "$29 trillion to $48 trillion" == "29-48 trillion USD"),
pick the most complete variant as canonical_value, every input variant must appear
in exactly one group.

## merge.py — semantic consensus merge (the "smart" part)

```python
async def merge_field(
    name: str,
    spec: FieldSpec,
    hits: list[FieldHit],          # all hits for this field across calls
    total_calls: int,
    merge_provider: OllamaProvider,
) -> ExtractedField
```

Algorithm:
1. Split hits into null hits and value hits.
2. Exact pre-grouping: normalize (strip, casefold, collapse whitespace) → group identical
   values, keep counts + groundings.
3. If ≤1 distinct normalized value → no LLM needed: build groups directly.
4. Else ONE LLM call: `merge_provider.structured(merge_prompt(...), MergeGroups)` to cluster
   the distinct raw variants semantically. Map each pre-group's count into its semantic
   group (sum counts of variants in a group). If LLM output loses/invents variants, fall
   back to exact pre-groups (log warning).
5. ValueGroup per cluster; winner = max count; score = winner.count / total_calls.
   Null hits form their own group with canonical_value None (can win if majority).

```python
async def merge_extractions(
    schema: OutputSchema,
    extractions: list[SchemaExtraction],
    merge_provider: OllamaProvider,
) -> list[ExtractedField]
```
For each field in schema (schema order): collect hits by field name (case-insensitive),
call merge_field. Fields with zero hits → ExtractedField(value=None, count=0, score=0.0,
candidates=[]). Ignore hits whose name is not in schema.

## pipeline.py

```python
class ExtractionConfig(BaseModel):
    calls_per_provider: int = 2
    timeout_s: float = 240.0

async def run_extraction(
    pdf_path: str,
    schema: OutputSchema,
    providers: list[OllamaProvider] | None = None,   # None → load_providers()
    config: ExtractionConfig | None = None,
) -> ExtractionResult
```
load PDF → fan_out extraction calls → merge_extractions (merge_provider = providers[0])
→ ExtractionResult.

## cli.py

```
python -m doc_extractor extract <pdf> --schema schema_sample.json \
    [--calls-per-provider 2] [--output result.json]
```
argparse, asyncio.run, pretty-print result JSON to stdout and optional file.
`__main__.py` delegates to `cli.main()`.

## schema_sample.json (test schema for the McKinsey PDF)

```json
{
  "schema_id": "mckinsey_report",
  "fields": [
    {"name": "title", "description": "Full title of the report", "type": "string"},
    {"name": "publisher", "description": "Organization that published the report", "type": "string"},
    {"name": "publication_date", "description": "Publication date of the report", "type": "string"},
    {"name": "authors", "description": "List of report authors, comma-separated", "type": "string"},
    {"name": "num_arenas", "description": "Number of future arenas of competition identified", "type": "integer"},
    {"name": "revenue_projection", "description": "Projected revenue range of the arenas by 2040", "type": "string"},
    {"name": "projection_year", "description": "Target year of the projections", "type": "integer"}
  ]
}
```

## Testing

- Unit tests: mock provider (class with scripted `structured()` responses); no network.
  Cover: loader (on the real PDF, pure local), fanout (success/partial failure/all-fail),
  merge (exact grouping, semantic fallback, null majority, zero hits), models round-trip.
- Integration: `tests/test_integration.py` marked `@pytest.mark.integration`
  (skipped unless `RUN_INTEGRATION=1`): run full pipeline on
  `the-next-big-arenas-of-competition-executive-summary-final.pdf` with schema_sample.json,
  assert result has all schema fields, title field score > 0.5.

## Conventions

- python >=3.11, type hints everywhere, pydantic v2 API only (model_validate, model_dump).
- logging via `logging.getLogger(__name__)`, no prints outside cli.
- every module ships with its unit tests in the same task.
- run tests: `uv run pytest -m "not integration"` (configure marker in pyproject).
