FROM python:3.13-slim-trixie

COPY --from=ghcr.io/astral-sh/uv:0.8.4 /uv /uvx /bin/

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1

ENV PYTHONUNBUFFERED=1

COPY pyproject.toml uv.lock /app/

RUN uv sync --frozen --no-cache --no-install-project

COPY . /app

RUN uv sync --frozen --no-cache

#CMD ["/app/.venv/bin/uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]