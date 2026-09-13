# =============================================================================
# Dockerfile — Production-style image for the PM2.5 Forecasting API
#
# What each section does:
#
#   FROM        → Base image (python:3.11-slim = small Debian + Python, no extras)
#   WORKDIR     → All subsequent commands run relative to /app inside the container
#   COPY reqs   → Copy requirements first (Docker layer cache trick — only
#                 reinstalls packages when requirements.txt changes)
#   RUN pip     → Install Python dependencies inside the image
#   COPY src    → Copy application code and config
#   EXPOSE      → Document that port 8000 will be used (doesn't actually open it)
#   CMD         → Default command to run when the container starts
# =============================================================================

FROM python:3.11-slim

# Prevents Python from writing .pyc files (keeps image clean)
ENV PYTHONDONTWRITEBYTECODE=1
# Prevents Python from buffering stdout/stderr (logs appear immediately)
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# ── Install dependencies (cached layer) ──────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ── Copy application code ─────────────────────────────────────────────────────
COPY config.yaml   .
COPY src/          src/
COPY api/          api/

# ── Copy trained artifacts (model weights + scaler) ──────────────────────────
# NOTE: The artifacts/ directory must exist locally with model.pt and scaler.pkl
# before running `docker build`.  See README for training instructions.
COPY artifacts/    artifacts/

# ── Expose the API port ───────────────────────────────────────────────────────
EXPOSE 8000

# ── Start the FastAPI server ──────────────────────────────────────────────────
# --host 0.0.0.0  → listen on all network interfaces (required inside Docker)
# --port 8000     → match the EXPOSE above
# --workers 1     → single worker; increase for production load
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
