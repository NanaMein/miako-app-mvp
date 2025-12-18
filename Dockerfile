FROM python:3.13-slim-trixie

COPY --from=ghcr.io/astral-sh/uv:0.8.4 /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock /app/

RUN uv sync --frozen --no-cache --no-install-project

COPY . /app

RUN uv sync --frozen --no-cache

CMD ["/app/.venv/bin/fastapi", "run", "main.py", "--port", "80", "--host", "0.0.0.0"]