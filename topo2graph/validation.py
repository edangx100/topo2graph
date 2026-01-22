"""
Topology validation with error and warning reporting.
"""

from typing import List, Tuple
from .schema import Topology, DeviceType


class ValidationError(Exception):
    """Raised when topology validation fails with blocking errors."""
    def __init__(self, errors: List[str]):
        self.errors = errors
        super().__init__("\n".join(errors))


class ValidationWarning:
    """Represents a non-blocking validation warning."""
    def __init__(self, message: str):
        self.message = message

    def __repr__(self) -> str:
        return f"ValidationWarning: {self.message}"


def validate_topology(topology: Topology) -> Tuple[List[str], List[ValidationWarning]]:
    """
    Validate a topology and return errors and warnings.

    Args:
        topology: The Topology object to validate

    Returns:
        Tuple of (errors, warnings). Empty list means no issues.

    Raises:
        ValidationError: If blocking validation errors are found
    """
    errors: List[str] = []
    warnings: List[ValidationWarning] = []

    # Check for empty topology
    if not topology.nodes:
        errors.append("Topology contains no nodes")
    if not topology.links:
        warnings.append(ValidationWarning("Topology contains no links"))

    # Check for unknown device types
    unknown_devices = [n for n in topology.nodes if n.type == DeviceType.UNKNOWN]
    if unknown_devices:
        device_list = ", ".join(n.label for n in unknown_devices)
        warnings.append(
            ValidationWarning(
                f"Devices with unknown type detected: {device_list}. "
                "This may indicate the vision model could not classify these devices."
            )
        )

    # Check for isolated nodes (no links)
    connected_node_ids = set()
    for link in topology.links:
        connected_node_ids.add(link.source)
        connected_node_ids.add(link.target)

    isolated_nodes = [
        n for n in topology.nodes
        if n.id not in connected_node_ids
    ]
    if isolated_nodes:
        device_list = ", ".join(n.label for n in isolated_nodes)
        warnings.append(
            ValidationWarning(
                f"Isolated nodes detected (no connections): {device_list}"
            )
        )

    # Check for missing recommended fields
    nodes_without_mgmt = [n for n in topology.nodes if not n.mgmt_ip]
    if nodes_without_mgmt:
        warnings.append(
            ValidationWarning(
                f"{len(nodes_without_mgmt)} node(s) missing management IP address"
            )
        )

    nodes_without_vendor = [n for n in topology.nodes if not n.vendor]
    if nodes_without_vendor:
        warnings.append(
            ValidationWarning(
                f"{len(nodes_without_vendor)} node(s) missing vendor information"
            )
        )

    # Raise if blocking errors found
    if errors:
        raise ValidationError(errors)

    return errors, warnings
