#!/usr/bin/env python3
import subprocess
import sys
import os


def run(cmd: list[str]):
    print(f"▶️ Running: {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, check=True)
    return result


if __name__ == "__main__":
    # Run Alembic migrations
    run([sys.executable, "-m", "alembic", "upgrade", "head"])

    # Start Uvicorn
    os.execvp(
        sys.executable,
        [
            sys.executable,
            "-m", "uvicorn",
            "src.main:app",
            "--host", "0.0.0.0",
            "--port", "8000",
        ]
    )