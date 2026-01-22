#!/usr/bin/env python3
"""
Topology extraction script using Hugging Face Inference API.

This script extracts network topology information from images using
the Qwen3-VL-8B-Instruct vision model via the HF Inference API.

Usage:
    python scripts/extract_topology_api.py --image <path_or_url>
    python scripts/extract_topology_api.py --image image/topo1.jpg --output output.json
    python scripts/extract_topology_api.py --image https://example.com/diagram.png
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from urllib.parse import urlparse

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from topo2graph.vision import TopologyExtractor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def is_url(path: str) -> bool:
    """Check if the given path is a URL."""
    parsed = urlparse(path)
    return parsed.scheme in ("http", "https")


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Extract network topology from diagram images using HF Inference API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/extract_topology_api.py --image image/topo1.jpg
  python scripts/extract_topology_api.py --image image/topo2.png --output result.json
  python scripts/extract_topology_api.py --image image/topo1.jpg --no-validate
        """
    )
    parser.add_argument(
        "--image", "-i",
        required=True,
        help="Path to local image file"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output file path (prints to stdout if not specified)"
    )
    parser.add_argument(
        "--api-key",
        help="HuggingFace API key (uses HF_TOKEN env var if not provided)"
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip schema validation (return raw extracted data)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress all logging output"
    )

    args = parser.parse_args()

    # Configure logging level
    if args.quiet:
        logging.disable(logging.CRITICAL)
    elif args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # URL support not yet implemented in TopologyExtractor
    if is_url(args.image):
        print("Error: URL input not yet supported. Please provide a local file path.",
              file=sys.stderr)
        sys.exit(1)

    try:
        extractor = TopologyExtractor(api_key=args.api_key)

        if args.no_validate:
            result = extractor.extract_from_image(args.image)
        else:
            topology = extractor.extract_topology(args.image)
            result = topology.model_dump()

        # Format output
        output_json = json.dumps(result, indent=2)

        if args.output:
            output_path = Path(args.output)
            output_path.write_text(output_json)
            logger.info(f"Output written to {output_path}")
            print(f"Topology extracted: {len(result.get('nodes', []))} nodes, "
                  f"{len(result.get('links', []))} links")
            print(f"Output saved to: {args.output}")
        else:
            print(output_json)

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Validation Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        logger.exception("Unexpected error")
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
