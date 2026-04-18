FROM python:3.13-slim-trixie

COPY --from=ghcr.io/astral-sh/uv:0.8.4 /uv /uvx /usr/local/bin/

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1

ENV PYTHONUNBUFFERED=1

COPY pyproject.toml uv.lock /app/

COPY . /app

RUN uv sync --frozen --no-cache \
    && rm -rf /root/.cache /tmp/* /var/tmp/* \
    && find /app -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

RUN chmod +x /app/execute.py

ENTRYPOINT ["/app/.venv/bin/python","/app/execute.py"]