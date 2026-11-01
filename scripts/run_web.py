from __future__ import annotations

import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
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


def open_ai_tunnel(destination: str) -> tuple[str, str, tempfile.TemporaryDirectory]:
    # The tunnel keeps OCRSD_AI_BASE_URL on loopback, so the remote server
    # needs no OCRSD_AI_ALLOW_REMOTE. ssh asks for the password in this
    # terminal, or uses SSH_ASKPASS when it is set.
    port = configured_port("OCRSD_AI_SSH_PORT", 11435)
    ensure_port(port)
    target = os.getenv("OCRSD_AI_SSH_TARGET", "127.0.0.1:11434")
    folder = tempfile.TemporaryDirectory(prefix="ocrsd-ai-ssh-")
    control = str(Path(folder.name) / "control")
    print(f"Abrindo o túnel SSH da IA: 127.0.0.1:{port} → {destination} ({target})", flush=True)
    opened = subprocess.run(["ssh", "-f", "-N", "-M", "-S", control,
                             "-o", "ExitOnForwardFailure=yes", "-o", "ConnectTimeout=15",
                             "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=3",
                             *os.getenv("OCRSD_AI_SSH_OPTIONS", "").split(),
                             "-L", f"127.0.0.1:{port}:{target}", destination])
    if opened.returncode != 0:
        folder.cleanup()
        raise SystemExit(f"Não foi possível abrir o túnel SSH até {destination}. Verifique a VPN e a senha.")
    os.environ["OCRSD_AI_BASE_URL"] = f"http://127.0.0.1:{port}/v1"
    return destination, control, folder


def close_ai_tunnel(tunnel: tuple[str, str, tempfile.TemporaryDirectory] | None) -> None:
    if tunnel is None:
        return
    destination, control, folder = tunnel
    subprocess.run(["ssh", "-S", control, "-O", "exit", destination], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    folder.cleanup()


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
    os.environ.setdefault("OCRSD_AI_PROVIDER", "openai")
    api_port = configured_port("OCRSD_API_PORT", 8000)
    ensure_port(api_port)
    destination = os.getenv("OCRSD_AI_SSH", "").strip()
    tunnel = open_ai_tunnel(destination) if destination else None
    try:
        return serve(api_port)
    finally:
        close_ai_tunnel(tunnel)


def serve(api_port: int) -> int:
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
        if os.getenv("OCRSD_AI_SSH"):
            print(f"IA em {os.environ['OCRSD_AI_BASE_URL']} (servidor {os.environ['OCRSD_AI_SSH']}, modelo {os.getenv('OCRSD_AI_MODEL', 'padrão')})", flush=True)
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
