FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY topo2graph/ ./topo2graph/
COPY templates/ ./templates/

# Expose port (Fly.io uses 8080 by default)
EXPOSE 8080

# Run the web server
CMD ["uvicorn", "topo2graph.web:app", "--host", "0.0.0.0", "--port", "8080"]
