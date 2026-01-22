"""
Main entry point for python -m topo2graph.cli
"""

from .cli import main
import sys

if __name__ == '__main__':
    sys.exit(main())
