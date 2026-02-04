# QDT Nexus - Multi-stage Dockerfile
# Stage 1: Build frontend
# Stage 2: Python backend with built frontend

# ============================================
# Stage 1: Build Next.js Frontend
# ============================================
FROM node:20-alpine AS frontend-builder

WORKDIR /app/frontend

# Copy package files
COPY frontend/package*.json ./

# Install dependencies
RUN npm ci --only=production=false

# Copy frontend source
COPY frontend/ ./

# Build Next.js app
RUN npm run build

# ============================================
# Stage 2: Python Backend
# ============================================
FROM python:3.11-slim AS production

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=5000 \
    HOST=0.0.0.0

# Create non-root user for security
RUN groupadd --gid 1000 appgroup && \
    useradd --uid 1000 --gid appgroup --shell /bin/bash --create-home appuser

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=appuser:appgroup *.py ./
COPY --chown=appuser:appgroup utils/ ./utils/
COPY --chown=appuser:appgroup configs/ ./configs/
COPY --chown=appuser:appgroup data/ ./data/

# Copy built frontend from Stage 1
COPY --from=frontend-builder --chown=appuser:appgroup /app/frontend/.next ./frontend/.next
COPY --from=frontend-builder --chown=appuser:appgroup /app/frontend/public ./frontend/public
COPY --from=frontend-builder --chown=appuser:appgroup /app/frontend/package.json ./frontend/

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/api/v1/health || exit 1

# Run the application
CMD ["python", "api_server.py"]
