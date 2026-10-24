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
from src.web.services import accepted_records, generate
from src.web.settings import Settings

logger = logging.getLogger("ocrsd.worker")


SUMMARY_FIELDS = ("method", "provider", "model", "registry_version", "analysis_id", "analysis_source", "context", "page",
                  "model_version", "mapping_revision", "pdf_sha256", "created_at", "reviewed", "assistant")


def base_provenance(row: dict) -> dict:
    return {"template_id": row["context"] or "modelo_web", "provider": "none", "model": None, "analysis_id": row["id"],
            "context": row["context"], "model_id": row["model_id"], "model_version": row["model_version"],
            "mapping_revision": row["mapping_revision"], "pdf_sha256": row["pdf_sha256"]}


def summary_of(provenance: dict) -> dict:
    return {name: provenance.get(name) for name in SUMMARY_FIELDS if name in provenance}


def process_analysis(store: JobStore, row: dict, provenance: dict | None = None) -> None:
    from src.mapping_assist.analysis import analyze_document
    from src.mapping_assist.matching import match_candidates
    from src.mapping_assist.proposal import build_proposal
    from src.mapping_assist.providers import ProviderStatus, assistant_from_settings

    identifier, settings = row["id"], store.settings
    provenance = base_provenance(row) if provenance is None else provenance
    store.update_analysis_progress(identifier, 15, "Extraindo texto e geometria")
    analysis = analyze_document(store.analysis_directory(identifier) / "template.pdf", page=row["page"] - 1)
    store.update_analysis_progress(identifier, 65, "Associando campos semânticos")
    definitions = store.list_custom_semantics(row["owner_id"])
    assistant = assistant_from_settings(settings, store.ai_cache)
    try:
        status = assistant.status()
    except Exception as exc:
        logger.warning("Assistente de IA indisponível para a análise %s: %s", identifier, exc)
        status = ProviderStatus(False, settings.ai_provider, settings.ai_model, f"Assistente indisponível: {str(exc) or type(exc).__name__}")
    report: dict = {}
    matches = match_candidates(analysis, row["context"], assistant if settings.ai_provider != "none" else None,
                               min_score=settings.mapping_min_score, min_margin=settings.mapping_min_margin,
                               custom_definitions=definitions, status=status, budget=settings.ai_budget, report=report)
    store.update_analysis_progress(identifier, 85, "Montando proposta para revisão")
    used = bool(status.available) and report.get("status") in {"used", "degraded"}
    warnings = list(report.get("warnings") or [])
    if settings.ai_provider != "none" and not status.available:
        warnings = [f"Assistente de IA indisponível ({status.detail}); proposta produzida por análise geométrica e regras semânticas."]
    provenance.update(provider=status.provider if used else "none", model=status.model if used else None,
                      assistant={"configured": settings.ai_provider, "status": report.get("status", "unavailable"),
                                 "detail": status.detail, "warnings": warnings})
    if warnings:
        provenance["warnings"] = warnings
    proposal = build_proposal(analysis, matches, row["context"], provenance, custom_definitions=definitions)
    store.finish_analysis(identifier, proposal=proposal, provenance=summary_of(proposal["provenance"]))


def execute_analysis(settings_data: dict, identifier: str, parent_pid: int) -> None:
    ctypes.CDLL(None).prctl(1, signal.SIGKILL)
    if os.getppid() != parent_pid:
        raise SystemExit(1)
    os.umask(0o077)
    settings = Settings(**settings_data)
    resource.setrlimit(resource.RLIMIT_AS, (settings.memory_mb * 1024**2,) * 2)
    resource.setrlimit(resource.RLIMIT_CPU, (settings.analysis_timeout, settings.analysis_timeout + 1))
    resource.setrlimit(resource.RLIMIT_FSIZE, (settings.max_job_bytes,) * 2)
    store = JobStore(settings)
    provenance = None
    try:
        import cv2
        cv2.setNumThreads(1)
        row = store.analysis_row(identifier)
        if row is None or row["status"] != "running":
            return
        provenance = base_provenance(row)
        process_analysis(store, row, provenance)
    except Exception as exc:
        logger.exception("Falha na análise assistida %s", identifier)
        store.finish_analysis(identifier, error="A análise excedeu o limite de memória do worker" if isinstance(exc, MemoryError)
                              else "Não foi possível analisar o PDF do modelo; confira o arquivo e tente novamente",
                              provenance=summary_of(provenance) if provenance else None)


AI_FAILURE = "A geração assistida por IA falhou:"


def failure_message(exc: Exception) -> str:
    from src.semantic.assisted import AssistedGenerationError

    if isinstance(exc, AssistedGenerationError):
        return f"{AI_FAILURE} {' '.join(str(exc).split())[:600]}"
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
    store = JobStore(settings)
    try:
        import cv2
        cv2.setNumThreads(1)
        row = store.get(identifier)
        options = GenerationOptions.model_validate_json(row["options"])
        byte_budget = settings.job_byte_budget(options.count)
        time_budget = settings.job_time_budget(options.count, options.generation_mode)
        resource.setrlimit(resource.RLIMIT_CPU, (time_budget, time_budget + 1))
        resource.setrlimit(resource.RLIMIT_FSIZE, (byte_budget,) * 2)
        replay = store.replay(row)
        if row.get("replay_of") and replay is None:
            raise ValueError("Os registros aceitos da geração original não estão disponíveis")
        artifacts = generate(options, store.directory(identifier), settings,
                             lambda completed: None, row.get("owner_id"),
                             lambda completed, pdf, ground_truth: store.publish_document(identifier, completed, pdf, ground_truth),
                             store.source(row), replay)
        if sum(item["size"] for item in artifacts) > byte_budget:
            raise ValueError("Limite de tamanho excedido")
        if options.generation_mode == "ai" and replay is None:
            store.save_records(identifier, accepted_records(store.directory(identifier)))
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
        generation_mode = json.loads(row["options"]).get("generation_mode", "traditional")
    except (TypeError, ValueError, AttributeError):
        generation_mode = "traditional"
    time_budget = store.settings.job_time_budget(row["total"], generation_mode)
    byte_budget = store.settings.job_byte_budget(row["total"])
    try:
        while process.is_alive():
            store.heartbeat()
            current = store.get(row["id"])
            size = sum(p.stat().st_size for p in store.directory(row["id"]).glob("*") if p.is_file())
            if current["cancel_requested"]:
                failure = "Cancelado"
            elif time.monotonic() - start > time_budget:
                failure = "Tempo máximo de execução excedido"
            elif size > byte_budget:
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


def run_analysis(store: JobStore, row: dict) -> None:
    process = multiprocessing.get_context("spawn").Process(target=execute_analysis, args=(store.settings.model_dump(), row["id"], os.getpid()))
    process.start()
    start = time.monotonic()
    failure = None
    try:
        while process.is_alive():
            store.heartbeat()
            if time.monotonic() - start > store.settings.analysis_timeout:
                failure = "Tempo máximo de análise excedido"
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
    current = store.analysis_row(row["id"])
    if current and current["status"] == "running":
        stopped = failure
        if stopped is None and process.exitcode is not None and process.exitcode < 0:
            stopped = "A análise excedeu um limite de recursos do servidor"
        store.finish_analysis(row["id"], error=stopped or "Processo de análise interrompido inesperadamente",
                              provenance=summary_of(base_provenance(current)))


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
            elif row := store.claim_analysis():
                logger.info("Executando análise %s", row["id"])
                run_analysis(store, row)
            if args.once:
                break
            if row is None:
                time.sleep(1)
        with store.connection() as db:
            db.execute("DELETE FROM worker")


if __name__ == "__main__":
    main()
