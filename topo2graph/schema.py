"""
Pydantic models for network topology validation.
"""

from enum import Enum
from ipaddress import IPv4Address, AddressValueError
from typing import Optional, List
from pydantic import BaseModel, Field, field_validator, model_validator


class DeviceType(str, Enum):
    """Allowed device types in the topology."""
    ROUTER = "router"
    SWITCH = "switch"
    FIREWALL = "firewall"
    SERVER = "server"
    WORKSTATION = "workstation"
    CLOUD = "cloud"
    UNKNOWN = "unknown"


class Node(BaseModel):
    """
    Represents a network device/node in the topology.

    Required fields: id, label, type
    Optional fields: mgmt_ip, vendor, model, role, site, zone, notes
    """
    id: str = Field(..., description="Unique identifier for the node")
    label: str = Field(..., description="Human-readable label")
    type: DeviceType = Field(..., description="Device type")
    mgmt_ip: Optional[str] = Field(None, description="Management IPv4 address")
    vendor: Optional[str] = Field(None, description="Device vendor")
    model: Optional[str] = Field(None, description="Device model")
    role: Optional[str] = Field(None, description="Device role in network")
    site: Optional[str] = Field(None, description="Site/location")
    zone: Optional[str] = Field(None, description="Network zone")
    notes: Optional[str] = Field(None, description="Additional notes")

    @field_validator('mgmt_ip')
    @classmethod
    def validate_ipv4(cls, v: Optional[str]) -> Optional[str]:
        """Validate IPv4 address format if provided."""
        if v is not None:
            try:
                IPv4Address(v)
            except AddressValueError as e:
                raise ValueError(f"Invalid IPv4 address: {v}") from e
        return v


class Link(BaseModel):
    """
    Represents a connection between two nodes in the topology.

    Required fields: source, target
    Optional fields: source_port, target_port
    """
    source: str = Field(..., description="Source node ID")
    target: str = Field(..., description="Target node ID")
    source_port: Optional[str] = Field(None, description="Source port name")
    target_port: Optional[str] = Field(None, description="Target port name")


class Topology(BaseModel):
    """
    Complete network topology with nodes and links.

    Validates:
    - Required fields on all nodes and links
    - Unique node IDs
    - Link referential integrity (source/target must exist)
    - IPv4 format validation
    """
    nodes: List[Node] = Field(default_factory=list, description="Network nodes")
    links: List[Link] = Field(default_factory=list, description="Network links")

    @model_validator(mode='after')
    def validate_topology(self) -> 'Topology':
        """Validate topology constraints."""
        # Check for unique node IDs
        node_ids = [node.id for node in self.nodes]
        unique_ids = set(node_ids)

        if len(node_ids) != len(unique_ids):
            # Find duplicates
            seen = set()
            duplicates = [nid for nid in node_ids if nid in seen or seen.add(nid)]
            raise ValueError(f"Duplicate node IDs found: {duplicates}")

        # Check link referential integrity
        node_id_set = set(node.id for node in self.nodes)
        for link in self.links:
            if link.source not in node_id_set:
                raise ValueError(f"Link source '{link.source}' does not exist in nodes")
            if link.target not in node_id_set:
                raise ValueError(f"Link target '{link.target}' does not exist in nodes")

        return self


def load_topology(json_data: dict) -> Topology:
    """
    Load and validate a topology from JSON data.

    Args:
        json_data: Dictionary containing topology JSON

    Returns:
        Validated Topology object

    Raises:
        ValueError: If validation fails
    """
    return Topology(**json_data)
