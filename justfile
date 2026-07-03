default:
    @just --list

# Extract sector forecasts from a PDF
run pdf="the-next-big-arenas-of-competition-executive-summary-final.pdf" *args="":
    uv run python -m doc_extractor extract {{pdf}} {{args}}

# Run test suite
test:
    uv run pytest

# Sync dependencies
sync:
    uv sync
