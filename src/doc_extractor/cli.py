from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from doc_extractor.loader import load_pdf
from doc_extractor.merge import merge_extractions
from doc_extractor.models import ExtractionResult, OutputSchema, SchemaExtraction
from doc_extractor.pipeline import ExtractionConfig, run_extraction
from doc_extractor.prompts import extraction_prompt
from doc_extractor.provider import OllamaProvider, load_providers
from doc_extractor.fanout import fan_out
from pydantic import BaseModel

logger = logging.getLogger(__name__)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="doc_extractor",
        description="Multi-call, multi-model structured document extraction",
    )
    subparsers = parser.add_subparsers(dest="command")
    extract_parser = subparsers.add_parser("extract", help="Extract fields from a PDF")
    extract_parser.add_argument("pdf", help="Path to PDF file")
    extract_parser.add_argument(
        "--schema",
        required=True,
        help="Path to OutputSchema JSON file",
    )
    extract_parser.add_argument(
        "--calls-per-provider",
        type=int,
        default=2,
        help="Number of extraction calls per provider (default: 2)",
    )
    extract_parser.add_argument(
        "--output",
        default=None,
        help="Optional path to write JSON result to",
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

    schema_path = Path(args.schema)
    if not schema_path.exists():
        logger.error("Schema file not found: %s", schema_path)
        sys.exit(1)

    schema_dict = json.loads(schema_path.read_text())
    schema = OutputSchema.model_validate(schema_dict)

    config = ExtractionConfig(calls_per_provider=args.calls_per_provider)

    result = asyncio.run(
        run_extraction(
            pdf_path=args.pdf,
            schema=schema,
            config=config,
        )
    )

    json_output = result.model_dump_json(indent=2)
    print(json_output)

    if args.output:
        Path(args.output).write_text(json_output)
        logger.info("Result written to %s", args.output)


if __name__ == "__main__":
    main()
