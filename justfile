default:
    @just --list

# Extract fields from a PDF using the sample schema
run pdf="the-next-big-arenas-of-competition-executive-summary-final.pdf" schema="schema_sample.json" *args="":
    uv run python -m doc_extractor extract {{pdf}} --schema {{schema}} {{args}}

# Run test suite
test:
    uv run pytest

# Sync dependencies
sync:
    uv sync
