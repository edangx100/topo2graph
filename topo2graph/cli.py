"""
Command-line interface for topo2graph.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional
import dotenv

from .schema import load_topology
from .validation import validate_topology, ValidationError
from .graph import Neo4jClient, Neo4jConfigError

dotenv.load_dotenv()



def cmd_ingest(path: str) -> int:
    """
    Ingest a topology JSON file, validate it, and print counts.

    Args:
        path: Path to the topology JSON file

    Returns:
        Exit code (0 for success, 1 for error)
    """
    file_path = Path(path)

    # Check if file exists
    if not file_path.exists():
        print(f"Error: File not found: {path}", file=sys.stderr)
        return 1

    # Check if file is readable
    if not file_path.is_file():
        print(f"Error: Not a file: {path}", file=sys.stderr)
        return 1

    # Read JSON file
    try:
        with open(file_path, 'r') as f:
            json_data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: Could not read file: {e}", file=sys.stderr)
        return 1

    # Parse and validate topology
    try:
        topology = load_topology(json_data)
    except ValueError as e:
        print(f"Validation Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: Unexpected error during parsing: {e}", file=sys.stderr)
        return 1

    # Run additional validation
    try:
        errors, warnings = validate_topology(topology)
    except ValidationError as e:
        print("Validation Errors (blocking):", file=sys.stderr)
        for error in e.errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    # Print results
    print(f"Successfully validated topology from: {path}")
    print(f"  Nodes: {len(topology.nodes)}")
    print(f"  Links: {len(topology.links)}")

    # Print warnings if any
    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(f"  - {warning.message}")

    return 0


def cmd_sync(path: str) -> int:
    """
    Sync a validated topology JSON file to Neo4j.

    Args:
        path: Path to the topology JSON file

    Returns:
        Exit code (0 for success, 1 for error)
    """
    file_path = Path(path)

    # Check if file exists
    if not file_path.exists():
        print(f"Error: File not found: {path}", file=sys.stderr)
        return 1

    # Check if file is readable
    if not file_path.is_file():
        print(f"Error: Not a file: {path}", file=sys.stderr)
        return 1

    # Read JSON file
    try:
        with open(file_path, 'r') as f:
            json_data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: Could not read file: {e}", file=sys.stderr)
        return 1

    # Parse and validate topology with schema
    try:
        topology = load_topology(json_data)
    except ValueError as e:
        print(f"Validation Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: Unexpected error during parsing: {e}", file=sys.stderr)
        return 1

    # Run custom validation
    try:
        errors, warnings = validate_topology(topology)
    except ValidationError as e:
        print("Validation Errors (blocking):", file=sys.stderr)
        for error in e.errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    # Print warnings if any
    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(f"  - {warning.message}")

    # Connect to Neo4j and sync
    try:
        with Neo4jClient() as client:
            # Verify connection first
            client.verify_connection()
            print(f"Connected to Neo4j")

            # Sync topology
            stats = client.sync_topology(topology)

            print(f"Sync complete:")
            print(f"  Nodes synced: {stats.nodes_synced}")
            print(f"  Links synced: {stats.links_synced}")
            print(f"  Elapsed: {stats.elapsed_seconds}s")
    except Neo4jConfigError as e:
        print(f"Neo4j Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: Sync failed: {e}", file=sys.stderr)
        return 1

    return 0


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Topo2Graph - Convert network topology images to validated graphs"
    )
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # ingest command
    ingest_parser = subparsers.add_parser(
        'ingest',
        help='Ingest and validate a topology JSON file'
    )
    ingest_parser.add_argument(
        'path',
        help='Path to the topology JSON file'
    )

    # sync command
    sync_parser = subparsers.add_parser(
        'sync',
        help='Sync a validated topology JSON file to Neo4j'
    )
    sync_parser.add_argument(
        'path',
        help='Path to the topology JSON file'
    )

    # Parse arguments
    args = parser.parse_args()

    # Dispatch command
    if args.command == 'ingest':
        return cmd_ingest(args.path)
    elif args.command == 'sync':
        return cmd_sync(args.path)
    else:
        parser.print_help()
        return 1


if __name__ == '__main__':
    sys.exit(main())
