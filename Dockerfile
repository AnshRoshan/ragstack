FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN uv sync --frozen --no-dev --compile-bytecode

# indexes, caches and HF model weights live on volumes
VOLUME ["/app/.ragstack", "/root/.cache"]

EXPOSE 8000

ENTRYPOINT ["uv", "run", "--no-sync", "ragstack"]
CMD ["serve", "--host", "0.0.0.0"]
