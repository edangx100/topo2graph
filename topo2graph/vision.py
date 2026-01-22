"""
Vision model integration for topology extraction from images.

Uses Qwen3-VL models via Hugging Face Inference API for image-to-JSON topology extraction.

Available models:
- 8B: Qwen/Qwen3-VL-8B-Instruct (faster, good for simple diagrams)
- 32B: Qwen/Qwen3-VL-32B-Instruct (slower, better for complex diagrams)
"""

import base64
import json
import logging
import os
import time
from pathlib import Path
from typing import Optional, Dict, Any, Literal

from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from PIL import Image

from .schema import Topology
from .prompt import get_extraction_prompt

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Model presets: (model_name, provider)
MODEL_PRESETS = {
    "8b": ("Qwen/Qwen3-VL-8B-Instruct", "novita"),
    "32b": ("Qwen/Qwen3-VL-32B-Instruct", "together"),
}

# Default model
DEFAULT_MODEL = "8b"
MODEL_NAME, PROVIDER = MODEL_PRESETS[DEFAULT_MODEL]


class TopologyExtractor:
    """
    Extract network topology from images using Hugging Face Inference API.

    Attributes:
        client: The Hugging Face InferenceClient
        model_name: The model being used
        provider: The inference provider
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        provider: str = PROVIDER,
        api_key: Optional[str] = None
    ):
        """
        Initialize the topology extractor.

        Args:
            model_name: HuggingFace model name to use
            provider: Inference provider (e.g., "novita", "together")
            api_key: Hugging Face API token (uses HF_TOKEN env var if not provided)
        """
        self.model_name = model_name
        self.provider = provider

        # Get API key from parameter or environment
        if api_key is None:
            api_key = os.environ.get("HF_TOKEN")
            if not api_key:
                raise ValueError(
                    "HF_TOKEN not found in environment. "
                    "Set it in .env file or export it."
                )

        self.client = InferenceClient(api_key=api_key, provider=provider)
        logger.info(f"Initialized TopologyExtractor with model: {model_name} (provider: {provider})")

    @classmethod
    def from_preset(
        cls,
        model: Literal["8b", "32b"] = DEFAULT_MODEL,
        api_key: Optional[str] = None
    ) -> "TopologyExtractor":
        """
        Create a TopologyExtractor using a model preset.

        Args:
            model: Model preset to use ("8b" or "32b")
                - "8b": Faster, good for simple diagrams (< 20 devices)
                - "32b": Slower, better for complex diagrams (20+ devices)
            api_key: Hugging Face API token (uses HF_TOKEN env var if not provided)

        Returns:
            Configured TopologyExtractor instance
        """
        if model not in MODEL_PRESETS:
            raise ValueError(f"Unknown model preset: {model}. Choose from: {list(MODEL_PRESETS.keys())}")

        model_name, provider = MODEL_PRESETS[model]
        return cls(model_name=model_name, provider=provider, api_key=api_key)

    def _image_to_data_url(self, image_path: Path) -> str:
        """
        Convert an image file to a data URL for the API.

        Args:
            image_path: Path to the image file

        Returns:
            Data URL string (e.g., "data:image/png;base64,...")
        """
        # Determine MIME type
        mime_type = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }.get(image_path.suffix.lower(), "image/png")

        # Read and encode image
        with open(image_path, "rb") as f:
            image_data = f.read()
        base64_data = base64.b64encode(image_data).decode("utf-8")

        return f"data:{mime_type};base64,{base64_data}"

    def _create_extraction_prompt(self) -> str:
        """
        Create the system prompt for topology extraction.

        Returns:
            Prompt string instructing the model to extract topology
        """
        return get_extraction_prompt()

    def extract_from_image(
        self,
        image_path: str | Path,
        prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extract topology from an image file using HF Inference API.

        Args:
            image_path: Path to the image file
            prompt: Optional custom prompt (uses default if not provided)

        Returns:
            Dictionary containing extracted topology data

        Raises:
            ValueError: If extraction fails
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise ValueError(f"Image file not found: {image_path}")

        logger.info(f"Extracting topology from {image_path}")

        # Convert image to data URL
        image_url = self._image_to_data_url(image_path)

        # Use default prompt if none provided
        if prompt is None:
            prompt = self._create_extraction_prompt()

        # Prepare messages for the model (align with reference/hf_inference_api.py)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url
                        }
                    }
                ]
            }
        ]

        # Retry settings for transient errors
        max_retries = 3
        base_delay = 2.0  # seconds
        retryable_codes = {500, 502, 503, 504}

        last_error = None
        for attempt in range(max_retries):
            try:
                # Call HF Inference API
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                )

                # Extract response text
                output_text = completion.choices[0].message.content
                logger.info("Model inference completed")

                # Parse JSON from output
                return self._parse_json_output(output_text)

            except Exception as e:
                last_error = e
                error_str = str(e)

                # Check if this is a retryable server error
                is_retryable = any(f"{code} " in error_str for code in retryable_codes)

                if is_retryable and attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(f"Retryable error on attempt {attempt + 1}/{max_retries}: {e}. "
                                   f"Retrying in {delay:.1f}s...")
                    time.sleep(delay)
                    continue

                # Non-retryable error or final attempt
                logger.error(f"Extraction failed: {e}")
                raise ValueError(f"Failed to extract topology: {e}") from e

        # Should not reach here, but just in case
        raise ValueError(f"Failed to extract topology after {max_retries} attempts: {last_error}")

    def _repair_json(self, json_str: str) -> str:
        """
        Attempt to repair common JSON errors from LLM output.

        Fixes:
        - Trailing commas before ] or }
        - Missing commas between elements
        - Single quotes to double quotes
        - Unescaped newlines in strings

        Args:
            json_str: Potentially malformed JSON string

        Returns:
            Repaired JSON string
        """
        import re

        repaired = json_str

        # Replace single quotes with double quotes (but not inside strings)
        # This is a simple heuristic - may not work for all cases
        repaired = re.sub(r"(?<![\\])\'", '"', repaired)

        # Remove trailing commas before ] or }
        repaired = re.sub(r',\s*([}\]])', r'\1', repaired)

        # Add missing commas between elements
        # Pattern: "value" followed by whitespace and "key": or { or [
        # e.g., "value"\n"key": -> "value",\n"key":
        repaired = re.sub(
            r'("|\d|true|false|null)\s*\n\s*(")',
            r'\1,\n\2',
            repaired
        )

        # Fix missing comma after } or ] followed by { or [ or "
        repaired = re.sub(r'([}\]])\s*\n\s*([{\[""])', r'\1,\n\2', repaired)

        # Fix missing comma after } followed by "key":
        repaired = re.sub(r'(\})\s+(")', r'\1,\n        \2', repaired)

        return repaired

    def _parse_json_output(self, output: str) -> Dict[str, Any]:
        """
        Parse JSON from model output with automatic repair for common errors.

        Args:
            output: Raw text output from the model

        Returns:
            Parsed JSON dictionary

        Raises:
            ValueError: If JSON parsing fails even after repair attempts
        """
        # Try to extract JSON from the output
        output = output.strip()

        # Look for JSON boundaries
        json_start = output.find("{")
        json_end = output.rfind("}")

        if json_start == -1 or json_end == -1 or json_start >= json_end:
            raise ValueError("No valid JSON found in model output")

        json_str = output[json_start:json_end + 1]

        # First attempt: parse as-is
        initial_error = None
        try:
            data = json.loads(json_str)
            logger.info("Successfully parsed JSON from model output")
            return self._clean_empty_strings(data)
        except json.JSONDecodeError as e:
            initial_error = e
            logger.warning(f"Initial JSON parse failed: {e}, attempting repair")

        # Second attempt: try to repair common issues
        try:
            repaired = self._repair_json(json_str)
            data = json.loads(repaired)
            logger.info("Successfully parsed JSON after repair")
            return self._clean_empty_strings(data)
        except json.JSONDecodeError as repair_error:
            # Log both errors for debugging
            logger.error(f"JSON repair failed: {repair_error}")
            raise ValueError(
                f"Failed to parse JSON: {initial_error}. "
                f"Repair attempt also failed: {repair_error}"
            ) from repair_error

    def _clean_empty_strings(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert empty strings to None for optional fields.

        Models sometimes return "" instead of null for optional fields.
        This causes validation errors for fields like mgmt_ip.

        Args:
            data: Parsed JSON dictionary

        Returns:
            Cleaned dictionary with empty strings converted to None
        """
        if "nodes" in data:
            for node in data["nodes"]:
                for key, value in node.items():
                    if value == "":
                        node[key] = None
        if "links" in data:
            for link in data["links"]:
                for key, value in link.items():
                    if value == "":
                        link[key] = None

        # Deduplicate nodes (VLMs sometimes generate duplicates)
        data = self._deduplicate_nodes(data)
        return data

    def _deduplicate_nodes(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Remove duplicate nodes, keeping the first occurrence.

        VLMs sometimes generate duplicate node IDs. This method removes
        duplicates while preserving the original order.

        Args:
            data: Parsed JSON dictionary with nodes

        Returns:
            Dictionary with duplicate nodes removed
        """
        if "nodes" not in data:
            return data

        seen_ids = set()
        unique_nodes = []
        duplicates = []

        for node in data["nodes"]:
            node_id = node.get("id")
            if node_id not in seen_ids:
                seen_ids.add(node_id)
                unique_nodes.append(node)
            else:
                duplicates.append(node_id)

        if duplicates:
            logger.warning(f"Removed {len(duplicates)} duplicate node(s): {duplicates}")

        data["nodes"] = unique_nodes
        return data

    def extract_topology(
        self,
        image_path: str | Path,
        prompt: Optional[str] = None
    ) -> Topology:
        """
        Extract and validate topology from an image.

        Args:
            image_path: Path to the image file
            prompt: Optional custom prompt

        Returns:
            Validated Topology object

        Raises:
            ValueError: If extraction or validation fails
        """
        data = self.extract_from_image(image_path, prompt)

        try:
            topology = Topology(**data)
            logger.info(f"Successfully extracted and validated topology with "
                       f"{len(topology.nodes)} nodes and {len(topology.links)} links")
            return topology
        except Exception as e:
            raise ValueError(f"Topology validation failed: {e}") from e


def extract_topology_from_image(
    image_path: str | Path,
    model_name: str = MODEL_NAME,
    provider: str = PROVIDER,
    api_key: Optional[str] = None
) -> Topology:
    """
    Convenience function to extract topology from an image.

    Args:
        image_path: Path to the image file
        model_name: Model name to use
        provider: Inference provider (e.g., "novita", "hf")
        api_key: Hugging Face API token

    Returns:
        Validated Topology object
    """
    extractor = TopologyExtractor(model_name=model_name, provider=provider, api_key=api_key)
    return extractor.extract_topology(image_path)


def _run_cli() -> int:
    """
    CLI entry point for python -m topo2graph.vision.

    This is the entry point when running the module directly. Intended usage:

        # Show help
        python -m topo2graph.vision --help

        # Extract topology to file
        python -m topo2graph.vision extract image/topo1.jpg --output output.json

        # Extract and pipe to other tools
        python -m topo2graph.vision extract image/topo1.jpg | python -m json.tool

    The CLI serves as a standalone tool for developers to extract topologies
    from images, either for testing or ad-hoc extraction tasks.

    Returns:
        Exit code (0 for success, 1 for error)
    """
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        prog="python -m topo2graph.vision",
        description="Extract network topology from diagram images using vision AI"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # extract command
    extract_parser = subparsers.add_parser(
        "extract",
        help="Extract topology from an image file"
    )
    extract_parser.add_argument(
        "image_path",
        help="Path to the image file"
    )
    extract_parser.add_argument(
        "--model", "-m",
        choices=["8b", "32b"],
        default=DEFAULT_MODEL,
        help="Model to use: 8b (faster, simple diagrams) or 32b (slower, complex diagrams). Default: %(default)s"
    )
    extract_parser.add_argument(
        "--output", "-o",
        help="Output file path (prints to stdout if not specified)"
    )
    extract_parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip schema validation (return raw extracted data)"
    )
    extract_parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    extract_parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress all logging output"
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    if args.command == "extract":
        return _cmd_extract(args)

    parser.print_help()
    return 1


def _cmd_extract(args) -> int:
    """
    Handle the extract command.

    Args:
        args: Parsed command line arguments

    Returns:
        Exit code (0 for success, 1 for error)
    """
    import sys

    # Configure logging level
    if args.quiet:
        logging.disable(logging.CRITICAL)
    elif args.verbose:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )
    else:
        logging.basicConfig(
            level=logging.WARNING,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )

    image_path = Path(args.image_path)

    # Check if file exists
    if not image_path.exists():
        print(f"Error: File not found: {args.image_path}", file=sys.stderr)
        return 1

    if not image_path.is_file():
        print(f"Error: Not a file: {args.image_path}", file=sys.stderr)
        return 1

    try:
        extractor = TopologyExtractor.from_preset(model=args.model)

        if args.no_validate:
            # Return raw extracted data without validation
            result = extractor.extract_from_image(image_path)
        else:
            # Extract and validate topology
            topology = extractor.extract_topology(image_path)
            result = topology.model_dump()

        # Format output
        output_json = json.dumps(result, indent=2)

        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(output_json)
            print(f"Topology extracted: {len(result.get('nodes', []))} nodes, "
                  f"{len(result.get('links', []))} links", file=sys.stderr)
            print(f"Output saved to: {args.output}", file=sys.stderr)
        else:
            print(output_json)

        return 0

    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        logger.exception("Unexpected error during extraction")
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(_run_cli())
