FROM python:3.13-slim-trixie

COPY --from=ghcr.io/astral-sh/uv:0.8.4 /uv /uvx /usr/local/bin/

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1

ENV PYTHONUNBUFFERED=1

COPY pyproject.toml uv.lock /app/

RUN uv sync --frozen --no-cache --no-install-project

COPY . /app

RUN chmod +x /app/entrypoint.sh

RUN uv sync --frozen --no-cache

ENTRYPOINT ["/app/entrypoint.sh"]