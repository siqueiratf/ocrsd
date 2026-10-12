from __future__ import annotations

import argparse
import ctypes
import fcntl
import json
import logging
import multiprocessing
import os
import resource
import shutil
import signal
import time

from src.web.jobs import JobStore
from src.web.models import GenerationOptions
from src.web.services import generate
from src.web.settings import Settings

logger = logging.getLogger("ocrsd.worker")


def failure_message(exc: Exception) -> str:
    if isinstance(exc, MemoryError):
        return "A geração excedeu o limite de memória do worker"
    if isinstance(exc, FileNotFoundError):
        return "Um arquivo necessário ao modelo de geração não foi encontrado"
    if isinstance(exc, ValueError):
        message = str(exc).replace("\n", " ").strip()
        if message:
            return f"A geração foi rejeitada pelo motor: {message[:300]}"
    return "A geração foi interrompida por uma falha interna"


def execute(settings_data: dict, identifier: str, parent_pid: int) -> None:
    ctypes.CDLL(None).prctl(1, signal.SIGKILL)
    if os.getppid() != parent_pid:
        raise SystemExit(1)
    os.umask(0o077)
    settings = Settings(**settings_data)
    resource.setrlimit(resource.RLIMIT_AS, (settings.memory_mb * 1024**2,) * 2)
    resource.setrlimit(resource.RLIMIT_CPU, (settings.job_timeout, settings.job_timeout + 1))
    resource.setrlimit(resource.RLIMIT_FSIZE, (settings.max_job_bytes,) * 2)
    store = JobStore(settings)
    try:
        import cv2
        cv2.setNumThreads(1)
        row = store.get(identifier)
        artifacts = generate(GenerationOptions.model_validate_json(row["options"]), store.directory(identifier), settings,
                             lambda completed: None, row.get("owner_id"),
                             lambda completed, pdf, ground_truth: store.publish_document(identifier, completed, pdf, ground_truth),
                             store.source(row))
        if sum(item["size"] for item in artifacts) > settings.max_job_bytes:
            raise ValueError("Limite de tamanho excedido")
        published = {item["name"]: item for item in store.artifacts(store.get(identifier))}
        store.finish(identifier, "succeeded", artifacts=[published.get(item["name"], item) for item in artifacts])
    except Exception as exc:
        logger.exception("Falha no job %s", identifier)
        store.finish(identifier, "failed", failure_message(exc))


def run_job(store: JobStore, row: dict) -> None:
    process = multiprocessing.get_context("spawn").Process(target=execute, args=(store.settings.model_dump(), row["id"], os.getpid()))
    process.start()
    start = time.monotonic()
    failure = None
    try:
        while process.is_alive():
            store.heartbeat()
            current = store.get(row["id"])
            size = sum(p.stat().st_size for p in store.directory(row["id"]).glob("*") if p.is_file())
            if current["cancel_requested"]:
                failure = "Cancelado"
            elif time.monotonic() - start > store.settings.job_timeout:
                failure = "Tempo máximo de execução excedido"
            elif size > store.settings.max_job_bytes:
                failure = "Limite de armazenamento do job excedido"
            elif shutil.disk_usage(store.root).free < store.settings.min_free_bytes:
                failure = "Espaço livre insuficiente"
            if failure:
                process.terminate()
                process.join(3)
                if process.is_alive():
                    process.kill()
                break
            process.join(0.5)
    finally:
        if process.is_alive():
            process.kill()
        process.join()
    current = store.get(row["id"])
    if failure or current["status"] == "running":
        stopped = failure
        if stopped is None and process.exitcode is not None and process.exitcode < 0:
            stopped = "O processo de geração excedeu um limite de recursos do servidor"
        store.finish(row["id"], "failed", stopped or "Processo de geração interrompido inesperadamente")
    current = store.get(row["id"])
    if current["status"] != "succeeded":
        directory = store.directory(row["id"])
        published = {item["name"] for item in json.loads(current["artifacts"])}
        if not published:
            shutil.rmtree(directory, ignore_errors=True)
        elif directory.exists():
            for path in directory.iterdir():
                if path.is_file() and path.name not in published:
                    path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description="Worker persistente OCRSD")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--health", action="store_true")
    args = parser.parse_args()
    settings = Settings()
    store = JobStore(settings)
    if args.health:
        raise SystemExit(0 if store.ready() else 1)
    logging.basicConfig(level=logging.INFO)
    os.umask(0o077)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    stopping = False

    def stop(signum, frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    with (store.root / "worker.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("Já existe um worker ativo neste armazenamento")
        store.recover()
        while not stopping:
            store.heartbeat()
            store.cleanup()
            row = store.claim()
            if row:
                logger.info("Executando job %s", row["id"])
                run_job(store, row)
            if args.once:
                break
            if row is None:
                time.sleep(1)
        with store.connection() as db:
            db.execute("DELETE FROM worker")


if __name__ == "__main__":
    main()
