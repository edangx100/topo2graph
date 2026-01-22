"""
FastAPI web UI for reviewing extracted topology JSON files.
"""

import json
import os
from datetime import datetime

from dotenv import load_dotenv

# Load environment variables from .env file (for NEO4J_* credentials, etc.)
load_dotenv()
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request, Form, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse, RedirectResponse, FileResponse
import logging
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError as PydanticValidationError

from .schema import Topology, load_topology
from .validation import validate_topology, ValidationError, ValidationWarning
from .graph import is_configured as neo4j_is_configured, Neo4jClient, Neo4jConfigError, SyncStats


def get_neo4j_status() -> dict:
    """
    Get Neo4j connection status for UI display.

    Returns:
        dict with 'status' ("not_configured", "connected", "error") and 'message'
    """
    if not neo4j_is_configured():
        return {
            "status": "not_configured",
            "message": "Neo4j not configured (set NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)"
        }

    try:
        with Neo4jClient() as client:
            client.verify_connection()
        return {
            "status": "connected",
            "message": "Connected to Neo4j"
        }
    except Neo4jConfigError as e:
        return {
            "status": "error",
            "message": str(e)
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Connection failed: {e}"
        }

# Directory paths
# BASE_DIR is the project root (parent of topo2graph/)
BASE_DIR = Path(__file__).parent.parent
TEMPLATES_DIR = BASE_DIR / "templates"

# DATA_DIR can be overridden via environment variable for cloud deployment (e.g., Fly.io volumes)
# Defaults to BASE_DIR for local development
DATA_DIR = Path(os.environ.get("DATA_DIR", str(BASE_DIR)))
INBOX_DIR = DATA_DIR / "inbox"
PROCESSED_DIR = DATA_DIR / "processed"
APPROVED_DIR = DATA_DIR / "approved"
IMAGE_DIR = DATA_DIR / "image"

# Ensure directories exist
for dir_path in [INBOX_DIR, PROCESSED_DIR, APPROVED_DIR]:
    dir_path.mkdir(exist_ok=True)

# FastAPI app
app = FastAPI(title="Topo2Graph Review UI", version="0.1.0")

# Templates
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Mount static files for images
app.mount("/image", StaticFiles(directory=str(IMAGE_DIR), check_dir=False), name="image")


def list_inbox_files() -> list[dict]:
    """List JSON files in inbox directory, excluding partial files and .ready markers."""
    files = []
    if not INBOX_DIR.exists():
        return files

    for path in INBOX_DIR.iterdir():
        # Skip non-JSON files and .ready markers
        if not path.name.endswith('.json') or path.name.endswith('.ready'):
            continue
        # Skip partial files (check for corresponding .ready marker that doesn't exist yet)
        ready_marker = path.with_suffix('.json.ready')
        if ready_marker.exists():
            # File is still being written, skip it
            continue

        stat = path.stat()
        files.append({
            'name': path.name,
            'size': stat.st_size,
            'modified': datetime.fromtimestamp(stat.st_mtime),
        })

    # Sort by modification time, newest first
    files.sort(key=lambda x: x['modified'], reverse=True)
    return files


def load_json_file(filepath: Path) -> tuple[dict, Optional[str]]:
    """Load JSON file and return (data, error)."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        return data, None
    except json.JSONDecodeError as e:
        return {}, f"Invalid JSON: {e}"
    except Exception as e:
        return {}, f"Error loading file: {e}"


def validate_json_data(json_data: dict) -> tuple[Topology | None, list[str], list[str]]:
    """
    Validate JSON data against schema and business rules.

    Returns:
        Tuple of (topology, errors, warnings)
    """
    errors = []
    warnings = []
    topology = None

    # First, validate against Pydantic schema
    try:
        topology = load_topology(json_data)
    except PydanticValidationError as e:
        for error in e.errors():
            loc = " -> ".join(str(x) for x in error['loc'])
            errors.append(f"{loc}: {error['msg']}")
        return None, errors, warnings
    except Exception as e:
        errors.append(str(e))
        return None, errors, warnings

    # Then run custom validation
    try:
        _, validation_warnings = validate_topology(topology)
        warnings = [w.message for w in validation_warnings]
    except ValidationError as e:
        errors.extend(e.errors)

    return topology, errors, warnings


def write_audit_log(filename: str, node_count: int, link_count: int, approved_name: str) -> None:
    """Append audit entry to audit.log."""
    audit_path = APPROVED_DIR / "audit.log"
    timestamp = datetime.now().isoformat()
    entry = f"{timestamp}\t{filename}\t{node_count} nodes\t{link_count} links\t-> {approved_name}\n"
    with open(audit_path, 'a') as f:
        f.write(entry)


@app.get("/", response_class=HTMLResponse)
async def list_files(request: Request):
    """List pending JSON files in inbox."""
    files = list_inbox_files()
    neo4j_status = get_neo4j_status()
    return templates.TemplateResponse(
        "list.html",
        {"request": request, "files": files, "neo4j_status": neo4j_status}
    )


@app.get("/review/{filename}", response_class=HTMLResponse)
async def review_file(request: Request, filename: str):
    """Display review page for a specific file."""
    filepath = INBOX_DIR / filename

    if not filepath.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")

    # Load JSON data
    json_data, load_error = load_json_file(filepath)

    if load_error:
        return templates.TemplateResponse(
            "review.html",
            {
                "request": request,
                "filename": filename,
                "json_text": "",
                "load_error": load_error,
                "errors": [],
                "warnings": [],
                "nodes": [],
                "links": [],
                "image_url": None,
                "neo4j_status": get_neo4j_status(),
            }
        )

    # Validate
    topology, errors, warnings = validate_json_data(json_data)

    # Extract nodes and links for display
    nodes = json_data.get('nodes', [])
    links = json_data.get('links', [])

    # Check for image
    image_url = None
    if 'image_data_url' in json_data and json_data['image_data_url']:
        image_url = json_data['image_data_url']
    elif 'image_path' in json_data and json_data['image_path']:
        # Convert to URL path
        image_path = json_data['image_path']
        if image_path.startswith('image/'):
            image_url = '/' + image_path
        else:
            image_url = '/image/' + Path(image_path).name

    return templates.TemplateResponse(
        "review.html",
        {
            "request": request,
            "filename": filename,
            "json_text": json.dumps(json_data, indent=2),
            "load_error": None,
            "errors": errors,
            "warnings": warnings,
            "nodes": nodes,
            "links": links,
            "image_url": image_url,
            "neo4j_status": get_neo4j_status(),
        }
    )


@app.post("/validate", response_class=HTMLResponse)
async def validate_json(request: Request, filename: str = Form(...), json_text: str = Form(...)):
    """Validate edited JSON and re-render review page."""
    # Parse JSON
    try:
        json_data = json.loads(json_text)
    except json.JSONDecodeError as e:
        return templates.TemplateResponse(
            "review.html",
            {
                "request": request,
                "filename": filename,
                "json_text": json_text,
                "load_error": f"Invalid JSON: {e}",
                "errors": [],
                "warnings": [],
                "nodes": [],
                "links": [],
                "image_url": None,
                "neo4j_status": get_neo4j_status(),
            }
        )

    # Validate
    topology, errors, warnings = validate_json_data(json_data)

    # Extract nodes and links
    nodes = json_data.get('nodes', [])
    links = json_data.get('links', [])

    # Check for image
    image_url = None
    if 'image_data_url' in json_data and json_data['image_data_url']:
        image_url = json_data['image_data_url']
    elif 'image_path' in json_data and json_data['image_path']:
        image_path = json_data['image_path']
        if image_path.startswith('image/'):
            image_url = '/' + image_path
        else:
            image_url = '/image/' + Path(image_path).name

    return templates.TemplateResponse(
        "review.html",
        {
            "request": request,
            "filename": filename,
            "json_text": json.dumps(json_data, indent=2),
            "load_error": None,
            "errors": errors,
            "warnings": warnings,
            "nodes": nodes,
            "links": links,
            "image_url": image_url,
            "neo4j_status": get_neo4j_status(),
        }
    )


@app.post("/approve", response_class=HTMLResponse)
async def approve_file(request: Request, filename: str = Form(...), json_text: str = Form(...)):
    """Approve a reviewed file: validate, write to approved/, move to processed/."""
    # Parse JSON
    try:
        json_data = json.loads(json_text)
    except json.JSONDecodeError as e:
        return templates.TemplateResponse(
            "review.html",
            {
                "request": request,
                "filename": filename,
                "json_text": json_text,
                "load_error": f"Invalid JSON: {e}",
                "errors": ["Cannot approve: Invalid JSON"],
                "warnings": [],
                "nodes": [],
                "links": [],
                "image_url": None,
                "neo4j_status": get_neo4j_status(),
            }
        )

    # Validate - must pass all blocking checks
    topology, errors, warnings = validate_json_data(json_data)

    if errors:
        return templates.TemplateResponse(
            "review.html",
            {
                "request": request,
                "filename": filename,
                "json_text": json.dumps(json_data, indent=2),
                "load_error": None,
                "errors": errors + ["Cannot approve: Fix errors first"],
                "warnings": warnings,
                "nodes": json_data.get('nodes', []),
                "links": json_data.get('links', []),
                "image_url": None,
                "neo4j_status": get_neo4j_status(),
            }
        )

    # Generate timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = Path(filename).stem
    approved_name = f"{base_name}_{timestamp}.json"
    approved_path = APPROVED_DIR / approved_name

    # Write to approved/ using atomic rename
    temp_path = approved_path.with_suffix('.tmp')
    try:
        with open(temp_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        temp_path.rename(approved_path)
    except Exception as e:
        if temp_path.exists():
            temp_path.unlink()
        return templates.TemplateResponse(
            "review.html",
            {
                "request": request,
                "filename": filename,
                "json_text": json.dumps(json_data, indent=2),
                "load_error": None,
                "errors": [f"Error writing approved file: {e}"],
                "warnings": warnings,
                "nodes": json_data.get('nodes', []),
                "links": json_data.get('links', []),
                "image_url": None,
                "neo4j_status": get_neo4j_status(),
            }
        )

    # Move original to processed/
    source_path = INBOX_DIR / filename
    processed_path = PROCESSED_DIR / filename
    try:
        if source_path.exists():
            source_path.rename(processed_path)
    except Exception as e:
        # Log but don't fail - the approval already succeeded
        pass

    # Write audit log
    node_count = len(json_data.get('nodes', []))
    link_count = len(json_data.get('links', []))
    write_audit_log(filename, node_count, link_count, approved_name)

    # Attempt Neo4j sync if configured.
    # This is a best-effort sync - errors don't block approval success.
    # Status is one of: "not_configured", "success", "error"
    neo4j_status = "not_configured"
    neo4j_error = None
    neo4j_stats = None  # SyncStats dataclass with nodes_synced, links_synced, elapsed_seconds

    if neo4j_is_configured():
        try:
            # Sync uses idempotent MERGE operations, safe to re-run
            with Neo4jClient() as client:
                neo4j_stats = client.sync_topology(topology)
            neo4j_status = "success"
        except Neo4jConfigError as e:
            # Config errors (bad URI scheme, auth failure, connection refused)
            neo4j_status = "error"
            neo4j_error = str(e)
        except Exception as e:
            # Unexpected errors (query failure, network issues, etc.)
            neo4j_status = "error"
            neo4j_error = f"Sync failed: {e}"

    # Redirect to success page or list
    return templates.TemplateResponse(
        "approved.html",
        {
            "request": request,
            "filename": filename,
            "approved_name": approved_name,
            "node_count": node_count,
            "link_count": link_count,
            "neo4j_status": neo4j_status,
            "neo4j_error": neo4j_error,
            "neo4j_stats": neo4j_stats,
        }
    )


# Allowed image extensions
ALLOWED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.gif', '.webp'}


@app.get("/upload", response_class=HTMLResponse)
async def upload_page(request: Request):
    """Display the image upload page."""
    return templates.TemplateResponse(
        "upload.html",
        {"request": request, "error": None, "success": None}
    )


@app.post("/extract", response_class=HTMLResponse)
async def extract_topology(
    request: Request,
    image: UploadFile = File(...),
    model: str = Form("8b")
):
    """
    Handle image upload and trigger topology extraction.

    1. Validates the uploaded file is an image
    2. Saves to image/ with timestamped filename
    3. Runs vision model extraction (8b or 32b)
    4. Saves JSON to inbox/ with image_path reference
    5. Redirects to review page
    """
    # Validate file extension
    ext = Path(image.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return templates.TemplateResponse(
            "upload.html",
            {
                "request": request,
                "error": f"Invalid file type: {ext}. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
                "success": None,
            }
        )

    # Generate timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    original_stem = Path(image.filename).stem
    # Sanitize filename (remove spaces and special chars)
    safe_stem = "".join(c if c.isalnum() or c in '-_' else '_' for c in original_stem)
    image_filename = f"{safe_stem}_{timestamp}{ext}"
    image_path = IMAGE_DIR / image_filename

    # Save uploaded image
    try:
        IMAGE_DIR.mkdir(exist_ok=True)
        contents = await image.read()
        with open(image_path, 'wb') as f:
            f.write(contents)
    except Exception as e:
        return templates.TemplateResponse(
            "upload.html",
            {
                "request": request,
                "error": f"Failed to save image: {e}",
                "success": None,
            }
        )

    # Run vision extraction
    try:
        from .vision import TopologyExtractor

        extractor = TopologyExtractor.from_preset(model=model)
        extracted_data = extractor.extract_from_image(image_path)

        # Add image_path to the extracted data
        extracted_data['image_path'] = f"image/{image_filename}"

    except ValueError as e:
        # Extraction failed but image was saved
        return templates.TemplateResponse(
            "upload.html",
            {
                "request": request,
                "error": f"Extraction failed: {e}. Image saved to {image_filename}.",
                "success": None,
            }
        )
    except Exception as e:
        logging.exception("Extraction error")
        return templates.TemplateResponse(
            "upload.html",
            {
                "request": request,
                "error": f"Extraction error: {e}",
                "success": None,
            }
        )

    # Save JSON to inbox/
    json_filename = f"{safe_stem}_{timestamp}.json"
    json_path = INBOX_DIR / json_filename

    try:
        # Use atomic write
        temp_path = json_path.with_suffix('.tmp')
        with open(temp_path, 'w') as f:
            json.dump(extracted_data, f, indent=2)
        temp_path.rename(json_path)
    except Exception as e:
        return templates.TemplateResponse(
            "upload.html",
            {
                "request": request,
                "error": f"Failed to save extracted JSON: {e}",
                "success": None,
            }
        )

    # Redirect to review page
    return RedirectResponse(
        url=f"/review/{json_filename}",
        status_code=303  # See Other - for POST->GET redirect
    )
