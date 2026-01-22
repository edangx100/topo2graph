"""
Neo4j sync module for topo2graph.

Provides idempotent upsert of Device nodes and CONNECTED_TO relationships
to Neo4j AuraDB with configuration via environment variables.
"""

import os
import time
from dataclasses import dataclass
from typing import Optional

from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, AuthError

from .schema import Topology, DeviceType


# Network equipment types - devices that have physical network ports.
# When these devices connect to other network equipment, both source_port
# and target_port are meaningful and stored in the relationship.
# Endpoints (SERVER, WORKSTATION, CLOUD, UNKNOWN) don't have network ports,
# so only the target_port (on the network equipment they connect to) is stored.
NETWORK_EQUIPMENT_TYPES = {DeviceType.ROUTER, DeviceType.SWITCH, DeviceType.FIREWALL}


class Neo4jConfigError(Exception):
    """Raised when Neo4j configuration is invalid or missing."""
    pass


@dataclass
class SyncStats:
    """Statistics returned from a sync operation."""
    nodes_synced: int
    links_synced: int
    elapsed_seconds: float


def get_neo4j_config() -> tuple[str, str, str, str]:
    """
    Read Neo4j configuration from environment variables.

    Returns:
        Tuple of (uri, user, password, database)

    Raises:
        Neo4jConfigError: If required env vars are missing or URI scheme is invalid
    """
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USER")
    password = os.getenv("NEO4J_PASSWORD")
    database = os.getenv("NEO4J_DATABASE", "neo4j")

    if not uri:
        raise Neo4jConfigError("NEO4J_URI environment variable is not set")
    if not user:
        raise Neo4jConfigError("NEO4J_USER environment variable is not set")
    if not password:
        raise Neo4jConfigError("NEO4J_PASSWORD environment variable is not set")

    # Enforce neo4j+s scheme for AuraDB TLS
    if not uri.startswith("neo4j+s://"):
        raise Neo4jConfigError(
            f"NEO4J_URI must use 'neo4j+s://' scheme for AuraDB TLS connections. "
            f"Got: {uri.split('://')[0] if '://' in uri else 'no scheme'}://"
        )

    return uri, user, password, database


def is_configured() -> bool:
    """Check if Neo4j environment variables are configured."""
    try:
        get_neo4j_config()
        return True
    except Neo4jConfigError:
        return False


class Neo4jClient:
    """
    Neo4j client wrapper for syncing topology data.

    Usage:
        with Neo4jClient() as client:
            stats = client.sync_topology(topology)
    """

    def __init__(self, uri: Optional[str] = None, user: Optional[str] = None,
                 password: Optional[str] = None, database: Optional[str] = None):
        """
        Initialize Neo4j client.

        If parameters are not provided, reads from environment variables.

        Args:
            uri: Neo4j URI (must use neo4j+s:// scheme)
            user: Neo4j username
            password: Neo4j password
            database: Neo4j database name (default: "neo4j")

        Raises:
            Neo4jConfigError: If configuration is invalid
        """
        if uri and user and password:
            self._uri = uri
            self._user = user
            self._password = password
            self._database = database or "neo4j"

            if not self._uri.startswith("neo4j+s://"):
                raise Neo4jConfigError(
                    f"URI must use 'neo4j+s://' scheme for AuraDB TLS connections. "
                    f"Got: {self._uri.split('://')[0] if '://' in self._uri else 'no scheme'}://"
                )
        else:
            self._uri, self._user, self._password, self._database = get_neo4j_config()

        self._driver = None

    def __enter__(self):
        """Open connection to Neo4j."""
        self._driver = GraphDatabase.driver(
            self._uri,
            auth=(self._user, self._password)
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close connection to Neo4j."""
        if self._driver:
            self._driver.close()

    def verify_connection(self) -> bool:
        """
        Verify that the Neo4j connection is working.

        Returns:
            True if connection is successful

        Raises:
            Neo4jConfigError: If connection fails
        """
        try:
            self._driver.verify_connectivity()
            return True
        except ServiceUnavailable as e:
            raise Neo4jConfigError(f"Cannot connect to Neo4j: {e}") from e
        except AuthError as e:
            raise Neo4jConfigError(f"Neo4j authentication failed: {e}") from e

    def ensure_constraints(self) -> None:
        """
        Ensure uniqueness constraint exists on Device.id.

        Creates the constraint if it doesn't exist.
        """
        query = """
        CREATE CONSTRAINT device_id_unique IF NOT EXISTS
        FOR (d:Device) REQUIRE d.id IS UNIQUE
        """
        with self._driver.session(database=self._database) as session:
            session.run(query)

    def sync_topology(self, topology: Topology) -> SyncStats:
        """
        Sync a topology to Neo4j using idempotent MERGE operations.

        Args:
            topology: Validated Topology object

        Returns:
            SyncStats with counts and timing
        """
        start_time = time.time()

        # Ensure constraints exist
        self.ensure_constraints()

        nodes_synced = 0
        links_synced = 0

        with self._driver.session(database=self._database) as session:
            # Sync nodes using MERGE (idempotent upsert)
            for node in topology.nodes:
                session.run(
                    """
                    MERGE (d:Device {id: $id})
                    SET d.label = $label,
                        d.type = $type,
                        d.mgmt_ip = $mgmt_ip,
                        d.vendor = $vendor,
                        d.model = $model,
                        d.role = $role,
                        d.site = $site,
                        d.zone = $zone,
                        d.notes = $notes,
                        d.synced_at = datetime()
                    """,
                    id=node.id,
                    label=node.label,
                    type=node.type.value,
                    mgmt_ip=node.mgmt_ip,
                    vendor=node.vendor,
                    model=node.model,
                    role=node.role,
                    site=node.site,
                    zone=node.zone,
                    notes=node.notes,
                )
                nodes_synced += 1

            # Build a lookup dictionary to quickly find each node's device type by ID.
            # This is used to determine how to handle port properties on relationships.
            node_types = {node.id: node.type for node in topology.nodes}

            # Sync links using MERGE (idempotent upsert).
            # IMPORTANT: MERGE only on node IDs, then SET port properties.
            # This ensures only ONE relationship per direction between any two nodes,
            # preventing duplicates when port values change between syncs.
            for link in topology.links:
                # Look up device types to determine port handling and directionality.
                source_type = node_types.get(link.source)
                target_type = node_types.get(link.target)
                source_is_network_equipment = source_type in NETWORK_EQUIPMENT_TYPES
                target_is_network_equipment = target_type in NETWORK_EQUIPMENT_TYPES

                if source_is_network_equipment and target_is_network_equipment:
                    # Both are network equipment (ROUTER, SWITCH, FIREWALL).
                    # Create BIDIRECTIONAL relationships since network links are two-way.
                    # Forward direction: source -> target
                    # Reverse direction: target -> source (with ports swapped)
                    # Example: switch1(Gi0/1) <-> switch2(Gi0/2)

                    # Forward relationship: source -> target
                    session.run(
                        """
                        MATCH (source:Device {id: $source_id})
                        MATCH (target:Device {id: $target_id})
                        MERGE (source)-[r:CONNECTED_TO]->(target)
                        SET r.source_port = COALESCE($source_port, ""),
                            r.target_port = COALESCE($target_port, ""),
                            r.synced_at = datetime()
                        """,
                        source_id=link.source,
                        target_id=link.target,
                        source_port=link.source_port,
                        target_port=link.target_port,
                    )

                    # Reverse relationship: target -> source (ports swapped)
                    session.run(
                        """
                        MATCH (source:Device {id: $source_id})
                        MATCH (target:Device {id: $target_id})
                        MERGE (target)-[r:CONNECTED_TO]->(source)
                        SET r.source_port = COALESCE($target_port, ""),
                            r.target_port = COALESCE($source_port, ""),
                            r.synced_at = datetime()
                        """,
                        source_id=link.source,
                        target_id=link.target,
                        source_port=link.source_port,
                        target_port=link.target_port,
                    )
                    # Count as 2 links since we created bidirectional relationships
                    links_synced += 2

                elif source_is_network_equipment:
                    # Source is network equipment, target is endpoint.
                    # Single direction with both ports (network equipment has port info).
                    # Example: switch(Fa0/1) -> server
                    session.run(
                        """
                        MATCH (source:Device {id: $source_id})
                        MATCH (target:Device {id: $target_id})
                        MERGE (source)-[r:CONNECTED_TO]->(target)
                        SET r.source_port = COALESCE($source_port, ""),
                            r.target_port = COALESCE($target_port, ""),
                            r.synced_at = datetime()
                        """,
                        source_id=link.source,
                        target_id=link.target,
                        source_port=link.source_port,
                        target_port=link.target_port,
                    )
                    links_synced += 1

                else:
                    # Source is an endpoint (SERVER, WORKSTATION, CLOUD, UNKNOWN).
                    # Only include target_port since endpoints don't have network ports -
                    # only the network equipment they connect to has a port.
                    # Example: server -> switch(Eth1/24)
                    session.run(
                        """
                        MATCH (source:Device {id: $source_id})
                        MATCH (target:Device {id: $target_id})
                        MERGE (source)-[r:CONNECTED_TO]->(target)
                        SET r.target_port = COALESCE($target_port, ""),
                            r.synced_at = datetime()
                        """,
                        source_id=link.source,
                        target_id=link.target,
                        target_port=link.target_port,
                    )
                    links_synced += 1

        elapsed = time.time() - start_time

        return SyncStats(
            nodes_synced=nodes_synced,
            links_synced=links_synced,
            elapsed_seconds=round(elapsed, 3)
        )


def sync_topology_from_dict(data: dict) -> SyncStats:
    """
    Convenience function to sync topology from a dictionary.

    Args:
        data: Dictionary containing topology JSON (nodes and links)

    Returns:
        SyncStats with counts and timing

    Raises:
        Neo4jConfigError: If Neo4j is not configured
        ValueError: If topology validation fails
    """
    from .schema import load_topology

    topology = load_topology(data)

    with Neo4jClient() as client:
        return client.sync_topology(topology)
