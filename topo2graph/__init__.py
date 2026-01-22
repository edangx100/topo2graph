"""
Topo2Graph - Convert network topology images to validated graphs.
"""

__version__ = "0.1.0"

from .graph import (
    Neo4jClient,
    Neo4jConfigError,
    SyncStats,
    is_configured,
    sync_topology_from_dict,
)
