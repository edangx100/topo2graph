"""
Prompt templates for topology extraction.
"""

EXTRACTION_PROMPT = """You are a network topology extraction expert. Analyze the provided network diagram image and extract the topology information into structured JSON format.

Extract ALL visible network devices (nodes) and connections (links) from the diagram.

For each node, provide:
- id: A unique identifier (use the label or IP address as ID if no explicit ID shown)
- label: The device name/label shown in the diagram (use IP address if no name shown)
- type: One of: router, switch, firewall, server, workstation, cloud, unknown
- mgmt_ip: Management IP address if shown (optional)
- vendor: Vendor name if shown (e.g., Cisco, Juniper) (optional)
- model: Model number if shown (optional)
- role: Device role if shown (e.g., core, access, distribution) (optional)
- site: Site/location if shown (optional)
- zone: Network zone if shown (e.g., DMZ, internal) (optional)
- notes: Any other relevant information (optional)

CRITICAL: Determine device type by the ICON SHAPE shown in the diagram, NOT by the device label or name.

Example: A device labeled "server" that has a cylindrical router icon with horizontal lines is a ROUTER, NOT a server. The word "server" in the label does not determine the type - only the icon shape does.

Device type guidance (identify ONLY by ICON SHAPE, ignore labels):
- router: Cylindrical icon with horizontal lines/layers (classic router shape). If you see this icon, it's a router regardless of what the label says.
- switch: Flat/rectangular icon with arrows or port indicators. 
- firewall: Brick wall pattern or shield symbol.
- server: Tower/box shape or rack-mount server shape (tall rectangle). Only use if icon is a tower/box, NOT because label says "server".
- workstation: PC, laptop, desktop, or monitor icon shape.
- cloud: Cloud shape icon.
- unknown: When icon shape is unclear.

For each link/connection:
- source: The source node ID
- target: The target node ID
- source_port: Port name on the SOURCE device (IMPORTANT: extract if visible)
- target_port: Port name on the TARGET device (IMPORTANT: extract if visible)

PORT EXTRACTION IS IMPORTANT - Look carefully for port labels near connection lines:
- Common formats: Fa0/1, Fa0/24, Gi0/1, GigabitEthernet0/1, eth0, Ethernet0/0, e0/0, S0/0/0
- Ports are usually shown as small text near where the line connects to a device
- For switches: FastEthernet (Fa), GigabitEthernet (Gi), or just port numbers
- For routers: Serial (S), Ethernet (E/eth), GigabitEthernet (Gi)
- If a connection shows "Fa0/1" on one end and "Gi0/2" on the other, capture both

Return ONLY valid JSON with this exact structure:
{
  "nodes": [
    {
      "id": "string",
      "label": "string",
      "type": "router|switch|firewall|server|workstation|cloud|unknown",
      "mgmt_ip": "string (optional)",
      "vendor": "string (optional)",
      "model": "string (optional)",
      "role": "string (optional)",
      "site": "string (optional)",
      "zone": "string (optional)",
      "notes": "string (optional)"
    }
  ],
  "links": [
    {
      "source": "node_id",
      "target": "node_id",
      "source_port": "Fa0/1 (extract if visible)",
      "target_port": "Gi0/24 (extract if visible)"
    }
  ]
}

Important:
- Extract ALL devices shown in the diagram, including end-user devices and those shown only as IP addresses
- Extract ALL devices providing services (DHCP, DNS, TFTP, AAA, etc.) - classify them by their icon shape, not their service label
- Devices represented only by an IP address should be extracted as nodes (use the IP as both id and label)
- IMPORTANT: Always identify device type by ICON SHAPE, not the label. A device labeled "AAA server" with a router icon must be type "router" with appropriate vendor/model
- Include ALL connections between devices
- IMPORTANT: Extract port names (Fa0/1, Gi0/1, etc.) from connection lines when visible - these appear as small labels near device connection points
- Use consistent node IDs (referenced in links must exist in nodes)
- If uncertain about device type, use "unknown"
- Return only the JSON, no additional text or explanation"""



EXTRACTION_PROMPT_V0 = """You are a network topology extraction expert. Analyze the provided network diagram image and extract the topology information into structured JSON format.

Extract ALL visible network devices (nodes) and connections (links) from the diagram.

For each node, provide:
- id: A unique identifier (use the label or IP address as ID if no explicit ID shown)
- label: The device name/label shown in the diagram (use IP address if no name shown)
- type: One of: router, switch, firewall, server, workstation, cloud, unknown
- mgmt_ip: Management IP address if shown (optional)
- vendor: Vendor name if shown (e.g., Cisco, Juniper) (optional)
- model: Model number if shown (optional)
- role: Device role if shown (e.g., core, access, distribution) (optional)
- site: Site/location if shown (optional)
- zone: Network zone if shown (e.g., DMZ, internal) (optional)
- notes: Any other relevant information (optional)

CRITICAL: Determine device type by the ICON SHAPE shown in the diagram, NOT by the device label or name.

Example: A device labeled "AAA/billing server" that has a cylindrical router icon with horizontal lines is a ROUTER (type: "router", vendor: "Cisco", model: "c7200"), NOT a server. The word "server" in the label does not determine the type - only the icon shape does.

Device type guidance (identify ONLY by ICON SHAPE, ignore labels):
- router: Cylindrical icon with horizontal lines/layers (classic Cisco router shape). If you see this icon, it's a router regardless of what the label says. Cisco routers include c7200, c6500, c3600, c2600.
- switch: Flat/rectangular icon with arrows or port indicators. Cisco switches include C2900, C2960, C3750.
- firewall: Brick wall pattern or shield symbol.
- server: Tower/box shape or rack-mount server shape (tall rectangle). Only use if icon is a tower/box, NOT because label says "server".
- workstation: PC, laptop, desktop, or monitor icon shape.
- cloud: Cloud shape icon.
- unknown: When icon shape is unclear.

For each link/connection:
- source: The source node ID
- target: The target node ID
- source_port: Port name on the SOURCE device (IMPORTANT: extract if visible)
- target_port: Port name on the TARGET device (IMPORTANT: extract if visible)

PORT EXTRACTION IS IMPORTANT - Look carefully for port labels near connection lines:
- Common formats: Fa0/1, Fa0/24, Gi0/1, GigabitEthernet0/1, eth0, Ethernet0/0, e0/0, S0/0/0
- Ports are usually shown as small text near where the line connects to a device
- For switches: FastEthernet (Fa), GigabitEthernet (Gi), or just port numbers
- For routers: Serial (S), Ethernet (E/eth), GigabitEthernet (Gi)
- If a connection shows "Fa0/1" on one end and "Gi0/2" on the other, capture both

Return ONLY valid JSON with this exact structure:
{
  "nodes": [
    {
      "id": "string",
      "label": "string",
      "type": "router|switch|firewall|server|workstation|cloud|unknown",
      "mgmt_ip": "string (optional)",
      "vendor": "string (optional)",
      "model": "string (optional)",
      "role": "string (optional)",
      "site": "string (optional)",
      "zone": "string (optional)",
      "notes": "string (optional)"
    }
  ],
  "links": [
    {
      "source": "node_id",
      "target": "node_id",
      "source_port": "Fa0/1 (extract if visible)",
      "target_port": "Gi0/24 (extract if visible)"
    }
  ]
}

Important:
- Extract ALL devices shown in the diagram, including end-user devices and those shown only as IP addresses
- Extract ALL devices providing services (DHCP, DNS, TFTP, AAA, billing, etc.) - classify them by their icon shape, not their service label
- Devices represented only by an IP address should be extracted as nodes (use the IP as both id and label)
- IMPORTANT: Always identify device type by ICON SHAPE, not the label. A device labeled "AAA server" or "billing server" with a router icon must be type "router" with appropriate vendor/model
- Include ALL connections between devices
- IMPORTANT: Extract port names (Fa0/1, Gi0/1, etc.) from connection lines when visible - these appear as small labels near device connection points
- Use consistent node IDs (referenced in links must exist in nodes)
- If uncertain about device type, use "unknown"
- Return only the JSON, no additional text or explanation"""


def get_extraction_prompt() -> str:
    """
    Get the topology extraction prompt.

    Returns:
        Prompt string for topology extraction
    """
    return EXTRACTION_PROMPT
