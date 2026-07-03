from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from doc_extractor.constants import (
    DEFAULT_CALLS_PER_PROVIDER,
    DEFAULT_MAX_PAGES,
    DEFAULT_TIMEOUT_S,
)
from doc_extractor.pipeline import ExtractionConfig, run_extraction

logger = logging.getLogger(__name__)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="doc_extractor",
        description="Multi-call, multi-model forecast extraction from documents",
    )
    subparsers = parser.add_subparsers(dest="command")
    extract_parser = subparsers.add_parser("extract", help="Extract sector forecasts from a PDF")
    extract_parser.add_argument("pdf", help="Path to PDF file")
    extract_parser.add_argument(
        "--calls-per-provider",
        type=int,
        default=DEFAULT_CALLS_PER_PROVIDER,
        help=f"Number of extraction calls per provider (default: {DEFAULT_CALLS_PER_PROVIDER})",
    )
    extract_parser.add_argument(
        "--output",
        default=None,
        help="Optional path to write JSON result to",
    )
    extract_parser.add_argument(
        "--max-pages",
        type=int,
        default=DEFAULT_MAX_PAGES,
        help=f"Max number of pages to render (default: {DEFAULT_MAX_PAGES})",
    )
    extract_parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_S,
        help=f"Per-call timeout in seconds (default: {DEFAULT_TIMEOUT_S:g})",
    )
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = create_parser()

    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()

    if args.command != "extract":
        parser.print_help()
        sys.exit(1)

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        logger.error("PDF file not found: %s", pdf_path)
        sys.exit(1)

    config = ExtractionConfig(
        calls_per_provider=args.calls_per_provider,
        max_pages=args.max_pages,
        timeout_s=args.timeout,
    )

    result = asyncio.run(
        run_extraction(
            pdf_path=args.pdf,
            config=config,
        )
    )

    if result.total_calls == 0:
        logger.error(
            "Extraction failed: no provider calls succeeded (all failed or timed out).",
        )
        sys.exit(1)

    json_output = result.model_dump_json(indent=2)
    print(json_output)

    if args.output:
        Path(args.output).write_text(json_output)
        logger.info("Result written to %s", args.output)


if __name__ == "__main__":
    main()
