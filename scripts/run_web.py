from __future__ import annotations

import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]


def ensure_port(port: int) -> None:
    try:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", port))
    except PermissionError as exc:
        raise SystemExit("O ambiente não permite abrir conexões locais para o servidor web.") from exc
    except OSError as exc:
        raise SystemExit(f"A porta local {port} já está em uso.") from exc


def configured_port(name: str, default: int) -> int:
    value = os.getenv(name, str(default))
    try:
        port = int(value)
    except ValueError as exc:
        raise SystemExit(f"{name} precisa ser uma porta numérica válida.") from exc
    if not 1 <= port <= 65535:
        raise SystemExit(f"{name} precisa estar entre 1 e 65535.")
    return port


def stop(processes: list[tuple[str, subprocess.Popen]]) -> None:
    for _, process in processes:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
    deadline = time.monotonic() + 8
    for _, process in processes:
        if process.poll() is not None:
            continue
        try:
            process.wait(max(0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def main() -> int:
    os.environ.setdefault("OCRSD_STANDARD_EDITS", "true")
    api_port = configured_port("OCRSD_API_PORT", 8000)
    ensure_port(api_port)
    environment = {**os.environ, "PYTHONUNBUFFERED": "1"}
    commands = [
        ("API", [sys.executable, "-m", "uvicorn", "src.web.api:create_app", "--factory", "--host", "127.0.0.1", "--port", str(api_port)]),
        ("Worker", [sys.executable, "-m", "src.web.worker"]),
    ]
    processes: list[tuple[str, subprocess.Popen]] = []
    try:
        for name, command in commands:
            processes.append((name, subprocess.Popen(command, cwd=ROOT, env=environment, start_new_session=True)))
        print(f"\nOCRSD Web disponível em http://127.0.0.1:{api_port}", flush=True)
        print(f"API em http://127.0.0.1:{api_port}/api/docs", flush=True)
        print("Pressione Ctrl+C para encerrar a API e o worker.\n", flush=True)
        while True:
            for name, process in processes:
                code = process.poll()
                if code is not None:
                    print(f"{name} foi encerrado com código {code}.", file=sys.stderr, flush=True)
                    return code or 1
            time.sleep(0.3)
    except KeyboardInterrupt:
        print("\nEncerrando OCRSD Web...", flush=True)
        return 0
    finally:
        stop(processes)


if __name__ == "__main__":
    raise SystemExit(main())
