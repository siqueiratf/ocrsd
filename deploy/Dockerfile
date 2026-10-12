FROM ghcr.io/astral-sh/uv:0.12.10 AS uv
FROM python:3.11-slim-bookworm
COPY --from=uv /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OCRSD_STORAGE=/data OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
RUN apt-get update && apt-get install --no-install-recommends -y tesseract-ocr tesseract-ocr-por && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --extra web --no-dev --no-install-project
COPY src ./src
COPY documents ./documents
COPY assets ./assets
RUN uv sync --frozen --extra web --no-dev && useradd --uid 10001 --create-home ocrsd && mkdir /data && chown 10001:10001 /data
USER 10001:10001
ENV PATH="/app/.venv/bin:$PATH"
CMD ["uvicorn", "src.web.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips", "172.30.85.2"]
