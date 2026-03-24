# ---- Build stage ----
FROM python:3.12-slim AS builder

WORKDIR /build

# Install build dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy only what's needed for dependency resolution first (cache-friendly)
COPY pyproject.toml ./
COPY swarmpr/ ./swarmpr/

# Build wheel
RUN pip wheel --no-cache-dir --wheel-dir /build/wheels -e .


# ---- Runtime stage ----
FROM python:3.12-slim AS runtime

LABEL maintainer="Allan Hall <jallanhall@gmail.com>"
LABEL description="SwarmPR — multi-agent pipeline for risk-reviewed pull requests"

# Security: run as non-root
RUN groupadd --gid 1000 swarmpr \
    && useradd --uid 1000 --gid swarmpr --create-home swarmpr

# Install git (required for GitPython operations)
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install pre-built wheels from builder stage
COPY --from=builder /build/wheels /tmp/wheels
RUN pip install --no-cache-dir /tmp/wheels/*.whl \
    && rm -rf /tmp/wheels

# Copy application code
COPY swarmpr/ ./swarmpr/
COPY config.example.yaml ./config.example.yaml
COPY demo/ ./demo/

# Switch to non-root user
USER swarmpr

# Default config location
ENV SWARMPR_CONFIG=/app/config.yaml

# Expose API port
EXPOSE 8000

# Health check for the API server
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import httpx; r = httpx.get('http://localhost:8000/health'); r.raise_for_status()" || exit 1

# Default: run the API server
# Override with: docker run swarmpr swarmpr run --task "..."
ENTRYPOINT ["swarmpr"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8000"]
