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

from src.web.models import GenerationOptions, JobDetail, JobResponse
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
                CREATE TABLE IF NOT EXISTS workspaces (
                    id TEXT PRIMARY KEY, token_hash TEXT NOT NULL, created_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS models (
                    id TEXT PRIMARY KEY, owner_id TEXT NOT NULL, label TEXT NOT NULL,
                    description TEXT NOT NULL, status TEXT NOT NULL, version INTEGER NOT NULL,
                    mapping_id TEXT NOT NULL, field_count INTEGER NOT NULL,
                    created_at REAL NOT NULL, updated_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS models_owner ON models(owner_id, updated_at);
                CREATE TABLE IF NOT EXISTS presets (
                    id TEXT PRIMARY KEY, owner_id TEXT NOT NULL, name TEXT NOT NULL,
                    writing TEXT NOT NULL, filter_preset TEXT NOT NULL, shareable INTEGER NOT NULL,
                    created_at REAL NOT NULL, updated_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS presets_owner ON presets(owner_id, updated_at);
                CREATE TABLE IF NOT EXISTS generation_configs (
                    id TEXT PRIMARY KEY, owner_id TEXT NOT NULL, name TEXT NOT NULL,
                    options TEXT NOT NULL, created_at REAL NOT NULL, updated_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS configs_owner ON generation_configs(owner_id, updated_at);
                CREATE TABLE IF NOT EXISTS shares (
                    id TEXT PRIMARY KEY, job_id TEXT NOT NULL, owner_id TEXT NOT NULL,
                    token_hash TEXT NOT NULL, allow_download INTEGER NOT NULL,
                    created_at REAL NOT NULL, expires_at REAL NOT NULL, revoked_at REAL
                );
                CREATE INDEX IF NOT EXISTS shares_owner ON shares(owner_id, created_at);
            """)
            columns = {row[1] for row in db.execute("PRAGMA table_info(jobs)")}
            if "owner_id" not in columns:
                db.execute("ALTER TABLE jobs ADD COLUMN owner_id TEXT")

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

    def create_workspace(self) -> tuple[str, str]:
        identifier = uuid4().hex
        secret = secrets.token_urlsafe(32)
        token = f"{identifier}.{secret}"
        with self.connection() as db:
            db.execute("INSERT INTO workspaces VALUES (?,?,?)", (identifier, hashlib.sha256(token.encode()).hexdigest(), time.time()))
        return identifier, token

    def workspace(self, token: str) -> str:
        identifier = token.partition(".")[0]
        if len(identifier) != 32:
            raise ServiceError(401, "workspace_unauthorized", "Espaço de trabalho não autorizado")
        with self.connection() as db:
            row = db.execute("SELECT token_hash FROM workspaces WHERE id=?", (identifier,)).fetchone()
        if not row or not secrets.compare_digest(row[0], hashlib.sha256(token.encode()).hexdigest()):
            raise ServiceError(401, "workspace_unauthorized", "Espaço de trabalho não autorizado")
        return identifier

    def submit(self, options: GenerationOptions, client: str, owner_id: str | None = None) -> tuple[dict, str]:
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
            db.execute("INSERT INTO jobs (id,token_hash,client,options,status,total,created_at,updated_at,expires_at,owner_id) VALUES (?,?,?,?,?,?,?,?,?,?)",
                       (identifier, hashlib.sha256(token.encode()).hexdigest(), client_hash, options.model_dump_json(), "queued", options.count, now, now,
                        now + self.settings.retention_hours * 3600, owner_id))
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

    def detail(self, row: dict) -> JobDetail:
        try:
            options = GenerationOptions.model_validate_json(row["options"])
        except ValueError:
            options = None
        return JobDetail(**self.response(row).model_dump(), name=options.name if options else "Geração sem configuração válida", options=options)

    def list_jobs(self, owner_id: str, query: str = "", status: str | None = None) -> list[dict]:
        clauses = ["owner_id=?"]
        values: list[object] = [owner_id]
        if status:
            clauses.append("status=?")
            values.append(status)
        with self.connection() as db:
            rows = db.execute(f"SELECT * FROM jobs WHERE {' AND '.join(clauses)} ORDER BY created_at DESC LIMIT 200", values).fetchall()
        details = [self.detail(dict(row)).model_dump() for row in rows]
        if query:
            needle = query.casefold()
            details = [item for item in details if needle in item["name"].casefold() or item["options"] and needle in item["options"]["document_type"].casefold()]
        return details

    def authorize_owner(self, row: dict, owner_id: str) -> dict:
        if row.get("owner_id") != owner_id:
            raise ServiceError(404, "not_found", "Job não encontrado")
        return row

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

    def model_directory(self, identifier: str) -> Path:
        if len(identifier) != 32 or any(character not in "0123456789abcdef" for character in identifier):
            raise ServiceError(404, "not_found", "Modelo não encontrado")
        return self.root / "models" / identifier

    def create_model(self, owner_id: str, label: str, description: str, mapping_id: str,
                     pdf: bytes, definition: dict) -> dict:
        identifier = uuid4().hex
        directory = self.model_directory(identifier)
        directory.mkdir(mode=0o700, parents=True)
        (directory / "template.pdf").write_bytes(pdf)
        (directory / f"{mapping_id}.json").write_text(json.dumps(definition, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        now = time.time()
        with self.connection() as db:
            db.execute("INSERT INTO models VALUES (?,?,?,?,?,?,?,?,?,?)",
                       (identifier, owner_id, label, description, "draft", 1, mapping_id, len(definition["fields"]), now, now))
        return self.model(identifier, owner_id)

    def model(self, identifier: str, owner_id: str | None = None) -> dict:
        with self.connection() as db:
            row = db.execute("SELECT * FROM models WHERE id=?", (identifier,)).fetchone()
        if not row or owner_id is None or row["owner_id"] != owner_id:
            raise ServiceError(404, "not_found", "Modelo não encontrado")
        return dict(row)

    def list_models(self, owner_id: str) -> list[dict]:
        with self.connection() as db:
            rows = db.execute("SELECT * FROM models WHERE owner_id=? ORDER BY updated_at DESC", (owner_id,)).fetchall()
        return [dict(row) for row in rows]

    def update_model(self, identifier: str, owner_id: str, label: str, description: str, status: str) -> dict:
        self.model(identifier, owner_id)
        with self.connection() as db:
            db.execute("UPDATE models SET label=?,description=?,status=?,updated_at=? WHERE id=?",
                       (label, description, status, time.time(), identifier))
        return self.model(identifier, owner_id)

    def update_mapping(self, identifier: str, owner_id: str, definition: dict) -> dict:
        model = self.model(identifier, owner_id)
        path = self.model_directory(identifier) / f'{model["mapping_id"]}.json'
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(definition, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
        with self.connection() as db:
            db.execute("UPDATE models SET field_count=?,version=version+1,updated_at=? WHERE id=?",
                       (len(definition["fields"]), time.time(), identifier))
        return self.model(identifier, owner_id)

    def delete_model(self, identifier: str, owner_id: str) -> None:
        self.model(identifier, owner_id)
        with self.connection() as db:
            used = db.execute("SELECT COUNT(*) FROM jobs WHERE owner_id=? AND json_extract(options,'$.document_type')=?", (owner_id, identifier)).fetchone()[0]
            if used:
                raise ServiceError(409, "model_in_use", "O modelo possui gerações associadas e não pode ser excluído")
            db.execute("DELETE FROM models WHERE id=?", (identifier,))
        shutil.rmtree(self.model_directory(identifier))

    def list_presets(self, owner_id: str) -> list[dict]:
        with self.connection() as db:
            rows = db.execute("SELECT * FROM presets WHERE owner_id=? ORDER BY updated_at DESC", (owner_id,)).fetchall()
        return [dict(row) for row in rows]

    def save_preset(self, owner_id: str, name: str, writing: dict, filter_preset: str,
                    shareable: bool, identifier: str | None = None) -> dict:
        now = time.time()
        identifier = identifier or uuid4().hex
        with self.connection() as db:
            existing = db.execute("SELECT owner_id,created_at FROM presets WHERE id=?", (identifier,)).fetchone()
            if existing and existing["owner_id"] != owner_id:
                raise ServiceError(404, "not_found", "Preset não encontrado")
            created = existing["created_at"] if existing else now
            db.execute("INSERT OR REPLACE INTO presets VALUES (?,?,?,?,?,?,?,?)",
                       (identifier, owner_id, name, json.dumps(writing), filter_preset, int(shareable), created, now))
        return self.preset(identifier, owner_id)

    def preset(self, identifier: str, owner_id: str) -> dict:
        with self.connection() as db:
            row = db.execute("SELECT * FROM presets WHERE id=? AND owner_id=?", (identifier, owner_id)).fetchone()
        if not row:
            raise ServiceError(404, "not_found", "Preset não encontrado")
        return dict(row)

    def delete_preset(self, identifier: str, owner_id: str) -> None:
        self.preset(identifier, owner_id)
        with self.connection() as db:
            db.execute("DELETE FROM presets WHERE id=?", (identifier,))

    def save_config(self, owner_id: str, name: str, options: GenerationOptions, identifier: str | None = None) -> dict:
        now = time.time()
        identifier = identifier or uuid4().hex
        with self.connection() as db:
            existing = db.execute("SELECT owner_id,created_at FROM generation_configs WHERE id=?", (identifier,)).fetchone()
            if existing and existing["owner_id"] != owner_id:
                raise ServiceError(404, "not_found", "Configuração não encontrada")
            created = existing["created_at"] if existing else now
            db.execute("INSERT OR REPLACE INTO generation_configs VALUES (?,?,?,?,?,?)",
                       (identifier, owner_id, name, options.model_dump_json(), created, now))
        return self.config(identifier, owner_id)

    def config(self, identifier: str, owner_id: str) -> dict:
        with self.connection() as db:
            row = db.execute("SELECT * FROM generation_configs WHERE id=? AND owner_id=?", (identifier, owner_id)).fetchone()
        if not row:
            raise ServiceError(404, "not_found", "Configuração não encontrada")
        result = dict(row)
        result.pop("owner_id")
        result["options"] = json.loads(result["options"])
        return result

    def list_configs(self, owner_id: str) -> list[dict]:
        with self.connection() as db:
            rows = db.execute("SELECT id FROM generation_configs WHERE owner_id=? ORDER BY updated_at DESC", (owner_id,)).fetchall()
        return [self.config(row[0], owner_id) for row in rows]

    def delete_config(self, identifier: str, owner_id: str) -> None:
        self.config(identifier, owner_id)
        with self.connection() as db:
            db.execute("DELETE FROM generation_configs WHERE id=?", (identifier,))

    def create_share(self, row: dict, owner_id: str, allow_download: bool, expires_in_hours: int) -> tuple[dict, str]:
        self.authorize_owner(row, owner_id)
        if row["status"] != "succeeded":
            raise ServiceError(409, "not_ready", "Apenas resultados concluídos podem ser compartilhados")
        identifier = uuid4().hex
        token = secrets.token_urlsafe(32)
        now = time.time()
        expires_at = min(row["expires_at"], now + expires_in_hours * 3600)
        with self.connection() as db:
            db.execute("INSERT INTO shares VALUES (?,?,?,?,?,?,?,NULL)",
                       (identifier, row["id"], owner_id, hashlib.sha256(token.encode()).hexdigest(), int(allow_download), now, expires_at))
        return self.share(identifier, token), token

    def share(self, identifier: str, token: str) -> dict:
        with self.connection() as db:
            row = db.execute("SELECT * FROM shares WHERE id=?", (identifier,)).fetchone()
        if not row or not secrets.compare_digest(row["token_hash"], hashlib.sha256(token.encode()).hexdigest()):
            raise ServiceError(404, "not_found", "Compartilhamento não encontrado")
        if row["revoked_at"] is not None:
            raise ServiceError(410, "share_revoked", "Este compartilhamento foi revogado")
        if row["expires_at"] < time.time():
            raise ServiceError(410, "expired", "Este compartilhamento expirou")
        return dict(row)

    def list_shares(self, job_id: str, owner_id: str) -> list[dict]:
        self.authorize_owner(self.get(job_id), owner_id)
        with self.connection() as db:
            rows = db.execute("SELECT * FROM shares WHERE job_id=? ORDER BY created_at DESC", (job_id,)).fetchall()
        return [dict(row) for row in rows]

    def revoke_share(self, identifier: str, owner_id: str) -> dict:
        with self.connection() as db:
            row = db.execute("SELECT * FROM shares WHERE id=? AND owner_id=?", (identifier, owner_id)).fetchone()
            if not row:
                raise ServiceError(404, "not_found", "Compartilhamento não encontrado")
            db.execute("UPDATE shares SET revoked_at=? WHERE id=?", (time.time(), identifier))
            return dict(db.execute("SELECT * FROM shares WHERE id=?", (identifier,)).fetchone())
