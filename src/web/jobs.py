from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import secrets
import shutil
import sqlite3
import time
from pathlib import Path
from uuid import uuid4

from src.web.models import GenerationOptions, JobResponse
from src.web.services import ServiceError
from src.web.settings import Settings


class JobStore:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.root = settings.storage.resolve()
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        (self.root / "jobs").mkdir(mode=0o700, exist_ok=True)
        with self.connection() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY, token_hash TEXT NOT NULL, client TEXT NOT NULL,
                    options TEXT NOT NULL, status TEXT NOT NULL, completed INTEGER NOT NULL DEFAULT 0,
                    total INTEGER NOT NULL, created_at REAL NOT NULL, updated_at REAL NOT NULL,
                    expires_at REAL NOT NULL, cancel_requested INTEGER NOT NULL DEFAULT 0,
                    error TEXT, artifacts TEXT NOT NULL DEFAULT '[]'
                );
                CREATE INDEX IF NOT EXISTS jobs_client_created ON jobs(client, created_at);
                CREATE TABLE IF NOT EXISTS worker (id INTEGER PRIMARY KEY, heartbeat REAL NOT NULL);
            """)

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.root / "jobs.sqlite3", timeout=15)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def ready(self) -> bool:
        with self.connection() as db:
            row = db.execute("SELECT heartbeat FROM worker WHERE id=1").fetchone()
        return bool(row and row[0] > time.time() - 15)

    def heartbeat(self) -> None:
        with self.connection() as db:
            db.execute("INSERT OR REPLACE INTO worker VALUES (1, ?)", (time.time(),))

    def submit(self, options: GenerationOptions, client: str) -> tuple[dict, str]:
        if not self.settings.accepting_jobs:
            raise ServiceError(503, "maintenance", "Novos jobs temporariamente suspensos")
        now = time.time()
        token = secrets.token_urlsafe(32)
        identifier = uuid4().hex
        client_hash = hashlib.sha256(client.encode()).hexdigest()
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            if db.execute("SELECT COUNT(*) FROM jobs WHERE status IN ('queued','running')").fetchone()[0] >= self.settings.max_pending:
                raise ServiceError(429, "queue_full", "Fila cheia; tente mais tarde")
            if db.execute("SELECT COUNT(*) FROM jobs WHERE client=? AND created_at>?", (client_hash, now - 3600)).fetchone()[0] >= self.settings.jobs_per_hour:
                raise ServiceError(429, "rate_limit", "Limite horário de jobs atingido")
            reserved = db.execute("SELECT COUNT(*) FROM jobs WHERE status != 'expired'").fetchone()[0]
            if (reserved + 1) * self.settings.max_job_bytes > self.settings.max_storage_bytes:
                raise ServiceError(429, "storage_quota", "Armazenamento reservado esgotado; aguarde a expiração")
            if shutil.disk_usage(self.root).free < self.settings.min_free_bytes + self.settings.max_job_bytes:
                raise ServiceError(503, "storage_full", "Espaço insuficiente para um novo job")
            db.execute("INSERT INTO jobs (id,token_hash,client,options,status,total,created_at,updated_at,expires_at) VALUES (?,?,?,?,?,?,?,?,?)",
                       (identifier, hashlib.sha256(token.encode()).hexdigest(), client_hash, options.model_dump_json(), "queued", options.count, now, now,
                        now + self.settings.retention_hours * 3600))
        return self.get(identifier), token

    def get(self, identifier: str) -> dict:
        with self.connection() as db:
            row = db.execute("SELECT * FROM jobs WHERE id=?", (identifier,)).fetchone()
        if not row:
            raise ServiceError(404, "not_found", "Job não encontrado")
        return dict(row)

    def authorize(self, identifier: str, token: str) -> dict:
        row = self.get(identifier)
        if not secrets.compare_digest(row["token_hash"], hashlib.sha256(token.encode()).hexdigest()):
            raise ServiceError(404, "not_found", "Job não encontrado")
        return row

    def response(self, row: dict) -> JobResponse:
        return JobResponse(**{key: row[key] for key in JobResponse.model_fields})

    def claim(self) -> dict | None:
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT id FROM jobs WHERE status='queued' ORDER BY created_at LIMIT 1").fetchone()
            if row is None:
                return None
            db.execute("UPDATE jobs SET status='running',updated_at=? WHERE id=?", (time.time(), row[0]))
        return self.get(row[0])

    def progress(self, identifier: str, completed: int) -> None:
        with self.connection() as db:
            db.execute("UPDATE jobs SET completed=?,updated_at=? WHERE id=? AND status='running'", (completed, time.time(), identifier))

    def finish(self, identifier: str, status: str, error: str | None = None, artifacts: list | None = None) -> None:
        now = time.time()
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT cancel_requested FROM jobs WHERE id=?", (identifier,)).fetchone()
            if row[0]:
                status, artifacts, error = "cancelled", [], None
            db.execute("UPDATE jobs SET status=?,error=?,artifacts=?,updated_at=?,expires_at=? WHERE id=?",
                       (status, error, json.dumps(artifacts or []), now, now + self.settings.retention_hours * 3600, identifier))

    def cancel(self, identifier: str) -> dict:
        with self.connection() as db:
            db.execute("UPDATE jobs SET cancel_requested=1, status=CASE WHEN status='queued' THEN 'cancelled' ELSE status END, updated_at=? WHERE id=? AND status IN ('queued','running')",
                       (time.time(), identifier))
        return self.get(identifier)

    def recover(self) -> None:
        with self.connection() as db:
            rows = db.execute("SELECT id FROM jobs WHERE status='running'").fetchall()
        for row in rows:
            self.finish(row[0], "failed", "Worker interrompido; envie um novo job com a mesma semente")
            shutil.rmtree(self.directory(row[0]), ignore_errors=True)

    def directory(self, identifier: str) -> Path:
        if len(identifier) != 32 or any(c not in "0123456789abcdef" for c in identifier):
            raise ServiceError(404, "not_found", "Job não encontrado")
        return self.root / "jobs" / identifier

    def cleanup(self) -> None:
        now = time.time()
        with self.connection() as db:
            rows = db.execute("SELECT id FROM jobs WHERE status IN ('succeeded','failed','cancelled','expired') AND expires_at<?", (now,)).fetchall()
            db.execute("UPDATE jobs SET status='expired',artifacts='[]' WHERE status IN ('succeeded','failed','cancelled') AND expires_at<?", (now,))
        for row in rows:
            shutil.rmtree(self.directory(row[0]), ignore_errors=True)
        with self.connection() as db:
            db.execute("DELETE FROM jobs WHERE status='expired' AND expires_at<?", (now - 86400,))

    def artifacts(self, row: dict) -> list[dict]:
        if row["status"] == "expired" or row["expires_at"] < time.time():
            raise ServiceError(410, "expired", "Resultados expirados")
        if row["status"] != "succeeded":
            raise ServiceError(409, "not_ready", "Resultados ainda não disponíveis")
        return json.loads(row["artifacts"])

    def artifact(self, row: dict, artifact_id: str) -> tuple[Path, dict]:
        for item in self.artifacts(row):
            if item["id"] == artifact_id:
                path = self.directory(row["id"]) / item["name"]
                if path.resolve().parent != self.directory(row["id"]).resolve() or path.is_symlink() or not path.is_file():
                    break
                return path, item
        raise ServiceError(404, "not_found", "Artefato não encontrado")
