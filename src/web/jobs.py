from __future__ import annotations

from contextlib import contextmanager
import hashlib
import itertools
import json
import os
import secrets
import shutil
import sqlite3
import threading
import time
from pathlib import Path
from uuid import uuid4

from src.semantic.custom import validate_custom_definition
from src.web.models import GenerationOptions, JobDetail, JobResponse
from src.web.services import ServiceError
from src.web.settings import Settings


SEED_SEQUENCE = itertools.count()
ANALYSIS_CONTEXTS = {"dnv", "certidao_nascimento", "declaracao_obito"}


def fresh_seed(db: sqlite3.Connection) -> int:
    for _ in range(64):
        material = ":".join(str(value) for value in (time.time_ns(), time.perf_counter_ns(), os.getpid(), threading.get_ident(),
                                                         next(SEED_SEQUENCE))).encode() + secrets.token_bytes(16)
        seed = int.from_bytes(hashlib.blake2b(material, digest_size=8).digest(), "big") % 2**32
        if not db.execute("SELECT 1 FROM jobs WHERE json_extract(options, '$.seed')=?", (seed,)).fetchone():
            return seed
    raise ServiceError(503, "seed_unavailable", "Não foi possível reservar uma semente única; tente novamente")


class JobStore:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.root = settings.storage.resolve()
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        (self.root / "jobs").mkdir(mode=0o700, exist_ok=True)
        (self.root / "templates").mkdir(mode=0o700, exist_ok=True)
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
                CREATE TABLE IF NOT EXISTS field_types (
                    id TEXT PRIMARY KEY, owner_id TEXT NOT NULL, slug TEXT NOT NULL, name TEXT NOT NULL,
                    description TEXT NOT NULL, base_type TEXT NOT NULL, config TEXT NOT NULL,
                    created_at REAL NOT NULL, updated_at REAL NOT NULL, UNIQUE(owner_id, slug)
                );
                CREATE TABLE IF NOT EXISTS custom_semantics (
                    id TEXT PRIMARY KEY, owner_id TEXT NOT NULL, key TEXT NOT NULL,
                    label TEXT NOT NULL, description TEXT NOT NULL, data_type TEXT NOT NULL,
                    generator TEXT NOT NULL, created_at REAL NOT NULL, updated_at REAL NOT NULL,
                    UNIQUE(owner_id, key)
                );
                CREATE INDEX IF NOT EXISTS custom_semantics_owner ON custom_semantics(owner_id, label);
                CREATE TABLE IF NOT EXISTS analyses (
                    id TEXT PRIMARY KEY, owner_id TEXT NOT NULL, model_id TEXT NOT NULL,
                    status TEXT NOT NULL, created_at REAL NOT NULL, updated_at REAL NOT NULL,
                    started_at REAL, finished_at REAL, error TEXT, result TEXT, provenance TEXT
                );
                CREATE INDEX IF NOT EXISTS analyses_model ON analyses(model_id, owner_id, created_at);
                CREATE INDEX IF NOT EXISTS analyses_queue ON analyses(status, created_at);
                CREATE TABLE IF NOT EXISTS preset_grants (
                    preset_id TEXT NOT NULL, workspace_id TEXT NOT NULL, created_at REAL NOT NULL,
                    PRIMARY KEY (preset_id, workspace_id)
                );
                CREATE TABLE IF NOT EXISTS generation_drafts (
                    owner_id TEXT PRIMARY KEY, options TEXT NOT NULL, updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS wizard_drafts (
                    id TEXT PRIMARY KEY, owner_id TEXT NOT NULL, state TEXT NOT NULL, updated_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS wizard_drafts_owner ON wizard_drafts(owner_id, updated_at);
            """)
            columns = {row[1] for row in db.execute("PRAGMA table_info(jobs)")}
            if "owner_id" not in columns:
                db.execute("ALTER TABLE jobs ADD COLUMN owner_id TEXT")
            if "preset" not in columns:
                db.execute("ALTER TABLE jobs ADD COLUMN preset TEXT")
            if "display_name" not in columns:
                db.execute("ALTER TABLE jobs ADD COLUMN display_name TEXT")
            if "started_at" not in columns:
                db.execute("ALTER TABLE jobs ADD COLUMN started_at REAL")
            if "finished_at" not in columns:
                db.execute("ALTER TABLE jobs ADD COLUMN finished_at REAL")
            if "source_mapping" not in columns:
                db.execute("ALTER TABLE jobs ADD COLUMN source_mapping TEXT")
            if "source_template" not in columns:
                db.execute("ALTER TABLE jobs ADD COLUMN source_template TEXT")
            for column in ("records", "replay_of"):
                if column not in columns:
                    db.execute(f"ALTER TABLE jobs ADD COLUMN {column} TEXT")
            analysis_columns = {row[1] for row in db.execute("PRAGMA table_info(analyses)")}
            for column, definition in (("page", "INTEGER NOT NULL DEFAULT 1"), ("context", "TEXT"), ("mapping_id", "TEXT NOT NULL DEFAULT 'default'"),
                                       ("model_version", "INTEGER"), ("mapping_revision", "TEXT"), ("pdf_sha256", "TEXT"),
                                       ("progress", "INTEGER NOT NULL DEFAULT 0"), ("stage", "TEXT NOT NULL DEFAULT ''"),
                                       ("expires_at", "REAL NOT NULL DEFAULT 0"), ("client", "TEXT")):
                if column not in analysis_columns:
                    db.execute(f"ALTER TABLE analyses ADD COLUMN {column} {definition}")
            preset_columns = {row[1] for row in db.execute("PRAGMA table_info(presets)")}
            for column, definition in (("description", "TEXT NOT NULL DEFAULT ''"), ("version", "INTEGER NOT NULL DEFAULT 1"),
                                       ("share_token_hash", "TEXT")):
                if column not in preset_columns:
                    db.execute(f"ALTER TABLE presets ADD COLUMN {column} {definition}")

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

    def keep_template(self, template: Path) -> str:
        content = template.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        target = self.root / "templates" / f"{digest}.pdf"
        if not target.is_file():
            temporary = target.with_suffix(f".{secrets.token_hex(6)}.tmp")
            temporary.write_bytes(content)
            temporary.replace(target)
        return digest

    def source(self, row: dict) -> dict | None:
        if not row.get("source_mapping") or not row.get("source_template"):
            return None
        template = self.root / "templates" / f'{row["source_template"]}.pdf'
        if not template.is_file():
            return None
        return {"definition": json.loads(row["source_mapping"]), "template": template}

    def save_records(self, identifier: str, records: dict) -> None:
        with self.connection() as db:
            db.execute("UPDATE jobs SET records=? WHERE id=?", (json.dumps(records, ensure_ascii=False), identifier))

    def records(self, row: dict) -> dict | None:
        try:
            records = json.loads(row["records"]) if row.get("records") else None
        except ValueError:
            return None
        return records if isinstance(records, dict) and isinstance(records.get("documents"), list) else None

    def replay(self, row: dict) -> dict | None:
        records = self.records(row) if row.get("replay_of") else None
        return {**records, "source_job": row["replay_of"]} if records else None

    def submit(self, options: GenerationOptions, client: str, owner_id: str | None = None, preset: dict | None = None,
               source: dict | None = None, replay: tuple[str, dict] | None = None) -> tuple[dict, str]:
        if not self.settings.accepting_jobs:
            raise ServiceError(503, "maintenance", "Novos jobs temporariamente suspensos")
        template_hash = self.keep_template(source["template"]) if source else None
        mapping = json.dumps(source["definition"], ensure_ascii=False) if source else None
        now = time.time()
        token = secrets.token_urlsafe(32)
        identifier = uuid4().hex
        client_hash = hashlib.sha256(client.encode()).hexdigest()
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            if self.pending(db) >= self.settings.max_pending:
                raise ServiceError(429, "queue_full", "Fila cheia; tente mais tarde")
            if db.execute("SELECT COUNT(*) FROM jobs WHERE client=? AND created_at>?", (client_hash, now - 3600)).fetchone()[0] >= self.settings.jobs_per_hour:
                raise ServiceError(429, "rate_limit", "Limite horário de jobs atingido")
            reserved, reserved_count = db.execute("SELECT COALESCE(SUM(total),0), COUNT(*) FROM jobs WHERE status != 'expired'").fetchone()
            reserved_budget = reserved_count * self.settings.max_job_bytes + reserved * self.settings.job_bytes_per_document
            new_budget = self.settings.job_byte_budget(options.count)
            if reserved_budget + new_budget + self.cache_bytes() > self.settings.max_storage_bytes:
                raise ServiceError(429, "storage_quota", "Armazenamento reservado esgotado; aguarde a expiração")
            if shutil.disk_usage(self.root).free < self.settings.min_free_bytes + new_budget:
                raise ServiceError(503, "storage_full", "Espaço insuficiente para um novo job")
            if options.seed is None:
                options = options.model_copy(update={"seed": fresh_seed(db)})
            db.execute("INSERT INTO jobs (id,token_hash,client,options,status,total,created_at,updated_at,expires_at,owner_id,preset,source_mapping,source_template,replay_of,records) "
                       "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                       (identifier, hashlib.sha256(token.encode()).hexdigest(), client_hash, options.model_dump_json(), "queued", options.count, now, now,
                        now + self.settings.retention_hours * 3600, owner_id, json.dumps(preset, ensure_ascii=False) if preset else None,
                        mapping, template_hash, replay[0] if replay else None, json.dumps(replay[1], ensure_ascii=False) if replay else None))
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
        return JobDetail(**self.response(row).model_dump(), name=row.get("display_name") or (options.name if options else "Geração sem configuração válida"), options=options)

    def rename_job(self, identifier: str, owner_id: str, name: str) -> dict:
        cleaned = name.strip()
        if not cleaned or len(cleaned) > 100 or any(ord(character) < 32 for character in cleaned):
            raise ServiceError(422, "validation_error", "Informe um nome de até 100 caracteres")
        with self.connection() as db:
            updated = db.execute("UPDATE jobs SET display_name=? WHERE id=? AND owner_id=?", (cleaned, identifier, owner_id))
            if not updated.rowcount:
                raise ServiceError(404, "not_found", "Job não encontrado")
        return self.get(identifier)

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

    def list_jobs_page(self, owner_id: str, query: str = "", status: str = "", document_type: str = "",
                       order: str = "recentes", page: int = 1, page_size: int = 25) -> tuple[list[dict], int]:
        clauses = ["owner_id=?"]
        values: list[object] = [owner_id]
        statuses = [item for item in status.split(",") if item]
        if statuses:
            clauses.append(f"status IN ({','.join('?' for _ in statuses)})")
            values.extend(statuses)
        if document_type:
            clauses.append("json_extract(options, '$.document_type')=?")
            values.append(document_type)
        if query:
            escaped = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            clauses.append("(COALESCE(display_name, json_extract(options, '$.name')) LIKE ? ESCAPE '\\' OR json_extract(options, '$.document_type') LIKE ? ESCAPE '\\' OR id LIKE ? ESCAPE '\\')")
            values.extend([f"%{escaped}%"] * 3)
        where = " AND ".join(clauses)
        sort = {"recentes": "created_at DESC", "antigos": "created_at ASC", "data_asc": "created_at ASC", "data_desc": "created_at DESC",
                "nome_asc": "COALESCE(display_name, json_extract(options, '$.name')) COLLATE NOCASE ASC",
                "nome_desc": "COALESCE(display_name, json_extract(options, '$.name')) COLLATE NOCASE DESC",
                "modelo_asc": "json_extract(options, '$.document_type') COLLATE NOCASE ASC",
                "modelo_desc": "json_extract(options, '$.document_type') COLLATE NOCASE DESC",
                "quantidade_asc": "total ASC", "quantidade_desc": "total DESC",
                "tempo_asc": "CASE WHEN started_at IS NULL THEN NULL WHEN finished_at IS NOT NULL THEN finished_at-started_at WHEN status='running' THEN ?-started_at ELSE NULL END ASC",
                "tempo_desc": "CASE WHEN started_at IS NULL THEN NULL WHEN finished_at IS NOT NULL THEN finished_at-started_at WHEN status='running' THEN ?-started_at ELSE NULL END DESC"}.get(order, "created_at DESC")
        with self.connection() as db:
            total = db.execute(f"SELECT COUNT(*) FROM jobs WHERE {where}", values).fetchone()[0]
            rows = db.execute(f"SELECT * FROM jobs WHERE {where} ORDER BY {sort}, id DESC LIMIT ? OFFSET ?",
                              [*values, *([time.time()] if order in {"tempo_asc", "tempo_desc"} else []), page_size, (max(1, page) - 1) * page_size]).fetchall()
        return [{**self.detail(dict(row)).model_dump(), "started_at": row["started_at"], "finished_at": row["finished_at"]} for row in rows], total

    def activity(self, owner_id: str, since: float) -> dict:
        with self.connection() as db:
            rows = db.execute("SELECT * FROM jobs WHERE owner_id=? AND (status IN ('queued','running') OR (status IN ('succeeded','failed') AND finished_at>=?)) "
                              "ORDER BY created_at DESC LIMIT 20", (owner_id, since)).fetchall()
        jobs = [{**self.detail(dict(row)).model_dump(), "finished_at": row["finished_at"]} for row in rows]
        return {"active": [job for job in jobs if job["status"] in {"queued", "running"}][:3],
                "finished": [job for job in jobs if job["status"] in {"succeeded", "failed"}][:3]}

    def job_document_types(self, owner_id: str) -> list[str]:
        with self.connection() as db:
            rows = db.execute("SELECT DISTINCT json_extract(options, '$.document_type') AS document_type FROM jobs WHERE owner_id=? ORDER BY document_type", (owner_id,)).fetchall()
        return [row["document_type"] for row in rows if row["document_type"]]

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
            now = time.time()
            db.execute("UPDATE jobs SET status='running',started_at=?,updated_at=? WHERE id=?", (now, now, row[0]))
        return self.get(row[0])

    def progress(self, identifier: str, completed: int) -> None:
        with self.connection() as db:
            db.execute("UPDATE jobs SET completed=?,updated_at=? WHERE id=? AND status='running'", (completed, time.time(), identifier))

    def publish_document(self, identifier: str, completed: int, pdf: Path, ground_truth: Path) -> None:
        directory = self.directory(identifier).resolve()
        items = []
        for path, media_type in ((pdf, "application/pdf"), (ground_truth, "application/json")):
            if path.resolve().parent != directory or path.is_symlink() or not path.is_file():
                raise ValueError("Artefato inválido")
            items.append({"id": f"document-{completed}-{path.suffix[1:]}", "name": path.name,
                          "media_type": media_type, "size": path.stat().st_size, "document_number": completed})
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT artifacts,status,completed FROM jobs WHERE id=?", (identifier,)).fetchone()
            if not row or row["status"] != "running" or completed != row["completed"] + 1:
                raise ValueError("Job indisponível para publicação")
            artifacts = json.loads(row["artifacts"]) + items
            db.execute("UPDATE jobs SET artifacts=?,completed=?,updated_at=? WHERE id=?",
                       (json.dumps(artifacts), completed, time.time(), identifier))

    def finish(self, identifier: str, status: str, error: str | None = None, artifacts: list | None = None) -> None:
        now = time.time()
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT cancel_requested,artifacts FROM jobs WHERE id=?", (identifier,)).fetchone()
            if row[0]:
                status, error = "cancelled", None
            db.execute("UPDATE jobs SET status=?,error=?,artifacts=?,updated_at=?,finished_at=?,expires_at=? WHERE id=?",
                       (status, error, json.dumps(artifacts if artifacts is not None else json.loads(row["artifacts"])), now, now,
                        now + self.settings.retention_hours * 3600, identifier))

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
            current = self.get(row[0])
            directory = self.directory(row[0])
            published = {item["name"] for item in json.loads(current["artifacts"])}
            if not published:
                shutil.rmtree(directory, ignore_errors=True)
            elif directory.exists():
                for path in directory.iterdir():
                    if path.is_file() and path.name not in published:
                        path.unlink()
        self.recover_analyses()

    @property
    def ai_cache(self) -> Path:
        return self.root / "ai-cache"

    def cache_bytes(self) -> int:
        return sum(path.stat().st_size for path in self.ai_cache.glob("*.json") if path.is_file()) if self.ai_cache.is_dir() else 0

    def cleanup_ai_cache(self) -> None:
        if not self.ai_cache.is_dir():
            return
        limit = time.time() - self.settings.retention_hours * 3600
        for path in self.ai_cache.iterdir():
            try:
                if path.is_file() and path.stat().st_mtime < limit:
                    path.unlink()
            except OSError:
                pass

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
        self.cleanup_analyses()
        self.cleanup_ai_cache()

    def artifacts(self, row: dict) -> list[dict]:
        if row["status"] == "expired" or row["expires_at"] < time.time():
            raise ServiceError(410, "expired", "Resultados expirados")
        if row["status"] not in {"succeeded", "running", "failed", "cancelled"}:
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
                     pdf: bytes, definition: dict, layout: dict | None = None) -> dict:
        identifier = uuid4().hex
        directory = self.model_directory(identifier)
        directory.mkdir(mode=0o700, parents=True)
        (directory / "template.pdf").write_bytes(pdf)
        if layout is not None:
            (directory / "layout.json").write_text(json.dumps(layout, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
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

    def analysis_directory(self, identifier: str) -> Path:
        if len(identifier) != 32 or any(character not in "0123456789abcdef" for character in identifier):
            raise ServiceError(404, "not_found", "Análise não encontrada")
        return self.root / "analyses" / identifier

    def analysis_context(self, model: dict) -> str | None:
        from src.semantic.resolve import document_context

        path = self.model_directory(model["id"]) / f'{model["mapping_id"]}.json'
        try:
            template_id = json.loads(path.read_text(encoding="utf-8")).get("template_id")
        except (OSError, ValueError, AttributeError):
            template_id = None
        return next((context for value in (template_id, model["label"])
                     if (context := document_context(value)) in ANALYSIS_CONTEXTS), None)

    def submit_analysis(self, model_id: str, owner_id: str, page: int = 1, client: str | None = None) -> dict:
        from src.mapping_assist.analysis import page_count

        if not self.settings.accepting_jobs:
            raise ServiceError(503, "maintenance", "Novas análises temporariamente suspensas")
        model = self.model(model_id, owner_id)
        if model["status"] != "draft":
            raise ServiceError(409, "model_published", "Duplique o modelo publicado antes de analisar o mapeamento")
        if isinstance(page, bool) or not isinstance(page, int) or page < 1:
            raise ServiceError(422, "invalid_page", "Informe uma página válida a partir de 1")
        template = self.model_directory(model_id) / "template.pdf"
        if not template.is_file() or template.is_symlink():
            raise ServiceError(422, "invalid_pdf", "Template PDF do modelo indisponível")
        if template.stat().st_size > self.settings.max_upload_bytes:
            raise ServiceError(413, "file_too_large", "Template PDF excede o limite de upload")
        try:
            pages = page_count(template)
        except Exception as exc:
            raise ServiceError(422, "invalid_pdf", "Template PDF inválido") from exc
        if page > pages:
            raise ServiceError(422, "invalid_page", f"O PDF tem apenas {pages} página(s)")
        identifier = uuid4().hex
        directory = self.analysis_directory(identifier)
        directory.mkdir(mode=0o700, parents=True)
        try:
            snapshot = directory / "template.pdf"
            shutil.copyfile(template, snapshot)
            pdf_sha256 = hashlib.sha256(snapshot.read_bytes()).hexdigest()
            mapping_path = self.model_directory(model_id) / f'{model["mapping_id"]}.json'
            mapping_revision = hashlib.sha256(mapping_path.read_bytes()).hexdigest() if mapping_path.is_file() else None
            now = time.time()
            client_hash = hashlib.sha256(client.encode()).hexdigest() if client else None
            context = self.analysis_context(model)
            with self.connection() as db:
                db.execute("BEGIN IMMEDIATE")
                active = db.execute("SELECT id FROM analyses WHERE model_id=? AND owner_id=? AND page=? AND status IN ('queued','running') "
                                    "ORDER BY created_at DESC LIMIT 1", (model_id, owner_id, page)).fetchone()
                if active is None:
                    if self.pending(db, "analyses") >= self.settings.max_pending_analyses:
                        raise ServiceError(429, "queue_full", "Fila de análises cheia; tente mais tarde")
                    recent = db.execute("SELECT COUNT(*) FROM analyses WHERE created_at>? AND (owner_id=? OR client=?)",
                                        (now - 3600, owner_id, client_hash)).fetchone()[0]
                    if recent >= self.settings.jobs_per_hour:
                        raise ServiceError(429, "rate_limit", "Limite horário de análises atingido")
                    db.execute("INSERT INTO analyses (id,owner_id,model_id,page,context,mapping_id,model_version,mapping_revision,pdf_sha256,"
                               "status,stage,created_at,updated_at,expires_at,client) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                               (identifier, owner_id, model_id, page, context, model["mapping_id"], model["version"],
                                mapping_revision, pdf_sha256, "queued", "Aguardando análise", now, now,
                                now + self.settings.retention_hours * 3600, client_hash))
        except Exception:
            shutil.rmtree(directory, ignore_errors=True)
            raise
        if active is not None:
            shutil.rmtree(directory, ignore_errors=True)
            return self.analysis(active["id"], owner_id)
        return self.analysis(identifier, owner_id)

    @staticmethod
    def pending(db: sqlite3.Connection, table: str = "jobs") -> int:
        return db.execute(f"SELECT COUNT(*) FROM {table} WHERE status IN ('queued','running')").fetchone()[0]

    @staticmethod
    def analysis_view(data: dict) -> dict:
        if data["status"] == "expired":
            raise ServiceError(410, "expired", "Resultado da análise expirado")
        return {"id": data["id"], "model_id": data["model_id"], "page": data["page"], "status": data["status"],
                "progress": data["progress"], "stage": data["stage"], "error": data["error"],
                "provenance": json.loads(data["provenance"]) if data["provenance"] else None,
                "proposal": json.loads(data["result"]) if data["result"] else None,
                "created_at": data["created_at"], "updated_at": data["updated_at"],
                "started_at": data["started_at"], "finished_at": data["finished_at"]}

    def analysis(self, identifier: str, owner_id: str) -> dict:
        with self.connection() as db:
            row = db.execute("SELECT * FROM analyses WHERE id=? AND owner_id=?", (identifier, owner_id)).fetchone()
        if row is None:
            raise ServiceError(404, "not_found", "Análise não encontrada")
        return self.analysis_view(dict(row))

    def latest_analysis(self, model_id: str, owner_id: str) -> dict | None:
        with self.connection() as db:
            row = db.execute("SELECT * FROM analyses WHERE model_id=? AND owner_id=? AND status!='expired' "
                             "ORDER BY created_at DESC LIMIT 1", (model_id, owner_id)).fetchone()
        return self.analysis_view(dict(row)) if row else None

    def analysis_row(self, identifier: str) -> dict | None:
        with self.connection() as db:
            row = db.execute("SELECT * FROM analyses WHERE id=?", (identifier,)).fetchone()
        return dict(row) if row else None

    def claim_analysis(self) -> dict | None:
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM analyses WHERE status='queued' ORDER BY created_at LIMIT 1").fetchone()
            if row is None:
                return None
            now = time.time()
            db.execute("UPDATE analyses SET status='running',progress=5,stage='Preparando PDF',started_at=?,updated_at=? WHERE id=?",
                       (now, now, row["id"]))
            return dict(db.execute("SELECT * FROM analyses WHERE id=?", (row["id"],)).fetchone())

    def update_analysis_progress(self, identifier: str, progress: int, stage: str) -> None:
        with self.connection() as db:
            db.execute("UPDATE analyses SET progress=?,stage=?,updated_at=? WHERE id=? AND status='running'",
                       (max(0, min(progress, 99)), stage[:100], time.time(), identifier))

    def finish_analysis(self, identifier: str, proposal: dict | None = None, error: str | None = None,
                        provenance: dict | None = None) -> None:
        now = time.time()
        failed = error is not None or proposal is None
        with self.connection() as db:
            db.execute("UPDATE analyses SET status=?,progress=100,stage=?,result=?,error=?,provenance=?,updated_at=?,finished_at=?,expires_at=? "
                       "WHERE id=? AND status='running'",
                       ("failed" if failed else "succeeded", "Falha na análise" if failed else "Proposta pronta para revisão",
                        None if failed else json.dumps(proposal, ensure_ascii=False),
                        (error or "A análise não pôde ser concluída") if failed else None,
                        json.dumps(provenance, ensure_ascii=False) if provenance else None,
                        now, now, now + self.settings.retention_hours * 3600, identifier))
        shutil.rmtree(self.analysis_directory(identifier), ignore_errors=True)

    def recover_analyses(self) -> None:
        with self.connection() as db:
            rows = db.execute("SELECT id FROM analyses WHERE status='running'").fetchall()
        for row in rows:
            self.finish_analysis(row[0], error="Worker interrompido; inicie uma nova análise")

    def cleanup_analyses(self) -> None:
        now = time.time()
        with self.connection() as db:
            rows = db.execute("SELECT id FROM analyses WHERE status IN ('succeeded','failed') AND expires_at<?", (now,)).fetchall()
            db.execute("UPDATE analyses SET status='expired',result=NULL WHERE status IN ('succeeded','failed') AND expires_at<?", (now,))
            db.execute("DELETE FROM analyses WHERE status='expired' AND expires_at<?", (now - 86400,))
        for row in rows:
            shutil.rmtree(self.analysis_directory(row["id"]), ignore_errors=True)

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

    def layout(self, identifier: str) -> dict | None:
        path = self.model_directory(identifier) / "layout.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None

    def save_layout(self, identifier: str, owner_id: str, layout: dict, pdf: bytes) -> None:
        self.model(identifier, owner_id)
        directory = self.model_directory(identifier)
        for name, content in (("template.pdf", pdf), ("layout.json", (json.dumps(layout, ensure_ascii=False, indent=2) + "\n").encode())):
            temporary = directory / f".{name}.tmp"
            temporary.write_bytes(content)
            temporary.replace(directory / name)

    def set_status(self, identifier: str, owner_id: str, status: str) -> dict:
        self.model(identifier, owner_id)
        with self.connection() as db:
            db.execute("UPDATE models SET status=?,updated_at=? WHERE id=?", (status, time.time(), identifier))
        return self.model(identifier, owner_id)

    def list_field_types(self, owner_id: str) -> list[dict]:
        with self.connection() as db:
            rows = db.execute("SELECT * FROM field_types WHERE owner_id=? ORDER BY name COLLATE NOCASE", (owner_id,)).fetchall()
        return [{**dict(row), "config": json.loads(row["config"])} for row in rows]

    def save_field_type(self, owner_id: str, slug: str, name: str, description: str, base_type: str, config: dict) -> dict:
        now = time.time()
        with self.connection() as db:
            if db.execute("SELECT 1 FROM field_types WHERE owner_id=? AND slug=?", (owner_id, slug)).fetchone():
                raise ServiceError(409, "field_type_exists", "Já existe um tipo personalizado com este identificador")
            identifier = uuid4().hex
            db.execute("INSERT INTO field_types VALUES (?,?,?,?,?,?,?,?,?)",
                       (identifier, owner_id, slug, name, description, base_type, json.dumps(config, ensure_ascii=False), now, now))
        return next(item for item in self.list_field_types(owner_id) if item["id"] == identifier)

    def delete_field_type(self, identifier: str, owner_id: str) -> None:
        with self.connection() as db:
            if not db.execute("DELETE FROM field_types WHERE id=? AND owner_id=?", (identifier, owner_id)).rowcount:
                raise ServiceError(404, "not_found", "Tipo personalizado não encontrado")

    def list_custom_semantics(self, owner_id: str) -> list[dict]:
        with self.connection() as db:
            rows = db.execute("SELECT * FROM custom_semantics WHERE owner_id=? ORDER BY label COLLATE NOCASE, key", (owner_id,)).fetchall()
        return [{**dict(row), "generator": json.loads(row["generator"]), "status": "approved"} for row in rows]

    def custom_semantic(self, owner_id: str, key: str) -> dict:
        with self.connection() as db:
            row = db.execute("SELECT * FROM custom_semantics WHERE owner_id=? AND key=?", (owner_id, key)).fetchone()
        if row is None:
            raise ServiceError(404, "not_found", "Definição semântica não encontrada")
        return {**dict(row), "generator": json.loads(row["generator"]), "status": "approved"}

    def save_custom_semantic(self, owner_id: str, key: str, label: str, description: str, data_type: str,
                             generator: dict) -> dict:
        try:
            definition = validate_custom_definition(key, label, description, data_type, generator)
        except (TypeError, ValueError) as exc:
            raise ServiceError(422, "invalid_custom_semantic", str(exc)) from exc
        identifier = uuid4().hex
        now = time.time()
        with self.connection() as db:
            try:
                db.execute("INSERT INTO custom_semantics VALUES (?,?,?,?,?,?,?,?,?)",
                           (identifier, owner_id, definition["key"], definition["label"], definition["description"],
                            definition["data_type"], json.dumps(definition["generator"], ensure_ascii=False), now, now))
            except sqlite3.IntegrityError as exc:
                raise ServiceError(409, "custom_semantic_exists", "Já existe uma definição semântica com este identificador") from exc
        return self.custom_semantic(owner_id, key)

    def update_custom_semantic(self, owner_id: str, key: str, label: str, description: str, data_type: str,
                               generator: dict) -> dict:
        try:
            definition = validate_custom_definition(key, label, description, data_type, generator)
        except (TypeError, ValueError) as exc:
            raise ServiceError(422, "invalid_custom_semantic", str(exc)) from exc
        with self.connection() as db:
            updated = db.execute("UPDATE custom_semantics SET label=?,description=?,data_type=?,generator=?,updated_at=? "
                                 "WHERE owner_id=? AND key=?",
                                 (definition["label"], definition["description"], definition["data_type"],
                                  json.dumps(definition["generator"], ensure_ascii=False), time.time(), owner_id, key))
            if not updated.rowcount:
                raise ServiceError(404, "not_found", "Definição semântica não encontrada")
        return self.custom_semantic(owner_id, key)

    def delete_custom_semantic(self, owner_id: str, key: str) -> None:
        with self.connection() as db:
            if not db.execute("DELETE FROM custom_semantics WHERE owner_id=? AND key=?", (owner_id, key)).rowcount:
                raise ServiceError(404, "not_found", "Definição semântica não encontrada")

    def delete_model(self, identifier: str, owner_id: str) -> None:
        self.model(identifier, owner_id)
        with self.connection() as db:
            used = db.execute("SELECT COUNT(*) FROM jobs WHERE owner_id=? AND json_extract(options,'$.document_type')=?", (owner_id, identifier)).fetchone()[0]
            if used:
                raise ServiceError(409, "model_in_use", "O modelo possui gerações associadas e não pode ser excluído")
            if db.execute("SELECT 1 FROM analyses WHERE model_id=? AND owner_id=? AND status IN ('queued','running')",
                          (identifier, owner_id)).fetchone():
                raise ServiceError(409, "analysis_in_progress", "Aguarde o fim da análise antes de excluir o modelo")
            analyses = [row[0] for row in db.execute("SELECT id FROM analyses WHERE model_id=? AND owner_id=?",
                                                     (identifier, owner_id)).fetchall()]
            db.execute("DELETE FROM analyses WHERE model_id=? AND owner_id=?", (identifier, owner_id))
            db.execute("DELETE FROM models WHERE id=?", (identifier,))
        shutil.rmtree(self.model_directory(identifier))
        for analysis_id in analyses:
            shutil.rmtree(self.analysis_directory(analysis_id), ignore_errors=True)

    def list_presets(self, owner_id: str) -> list[dict]:
        with self.connection() as db:
            rows = db.execute("SELECT * FROM presets WHERE owner_id=? ORDER BY updated_at DESC", (owner_id,)).fetchall()
        return [dict(row) for row in rows]

    def save_preset(self, owner_id: str, name: str, writing: dict, filter_preset: str,
                    shareable: bool, identifier: str | None = None, description: str | None = None) -> dict:
        now = time.time()
        identifier = identifier or uuid4().hex
        with self.connection() as db:
            existing = db.execute("SELECT * FROM presets WHERE id=?", (identifier,)).fetchone()
            if existing and existing["owner_id"] != owner_id:
                raise ServiceError(404, "not_found", "Preset não encontrado")
            if existing:
                db.execute("UPDATE presets SET name=?,writing=?,filter_preset=?,shareable=?,description=?,version=version+1,updated_at=?,"
                           "share_token_hash=CASE WHEN ? THEN share_token_hash ELSE NULL END WHERE id=?",
                           (name, json.dumps(writing), filter_preset, int(shareable), existing["description"] if description is None else description,
                            now, int(shareable), identifier))
                if not shareable:
                    db.execute("DELETE FROM preset_grants WHERE preset_id=?", (identifier,))
            else:
                db.execute("INSERT INTO presets (id,owner_id,name,writing,filter_preset,shareable,created_at,updated_at,description,version) "
                           "VALUES (?,?,?,?,?,?,?,?,?,1)",
                           (identifier, owner_id, name, json.dumps(writing), filter_preset, int(shareable), now, now, description or ""))
        return self.preset(identifier, owner_id)

    def preset(self, identifier: str, owner_id: str) -> dict:
        with self.connection() as db:
            row = db.execute("SELECT * FROM presets WHERE id=? AND owner_id=?", (identifier, owner_id)).fetchone()
        if not row:
            raise ServiceError(404, "not_found", "Preset não encontrado")
        return dict(row)

    def accessible_preset(self, identifier: str, workspace_id: str) -> tuple[dict, str]:
        with self.connection() as db:
            row = db.execute("SELECT * FROM presets WHERE id=?", (identifier,)).fetchone()
            granted = row is not None and db.execute("SELECT 1 FROM preset_grants WHERE preset_id=? AND workspace_id=?",
                                                     (identifier, workspace_id)).fetchone() is not None
        if row and row["owner_id"] == workspace_id:
            return dict(row), "meu"
        if row and granted and row["share_token_hash"]:
            return dict(row), "compartilhado"
        raise ServiceError(404, "not_found", "Preset não encontrado")

    def list_shared_presets(self, workspace_id: str) -> list[dict]:
        with self.connection() as db:
            rows = db.execute("SELECT presets.* FROM presets JOIN preset_grants ON preset_grants.preset_id = presets.id "
                              "WHERE preset_grants.workspace_id=? AND presets.share_token_hash IS NOT NULL ORDER BY presets.updated_at DESC",
                              (workspace_id,)).fetchall()
        return [dict(row) for row in rows]

    def share_preset(self, identifier: str, owner_id: str) -> str:
        self.preset(identifier, owner_id)
        token = secrets.token_urlsafe(32)
        with self.connection() as db:
            db.execute("UPDATE presets SET shareable=1,share_token_hash=? WHERE id=?", (hashlib.sha256(token.encode()).hexdigest(), identifier))
        return token

    def unshare_preset(self, identifier: str, owner_id: str) -> None:
        self.preset(identifier, owner_id)
        with self.connection() as db:
            db.execute("UPDATE presets SET shareable=0,share_token_hash=NULL WHERE id=?", (identifier,))
            db.execute("DELETE FROM preset_grants WHERE preset_id=?", (identifier,))

    def accept_preset(self, identifier: str, token: str, workspace_id: str) -> dict:
        with self.connection() as db:
            row = db.execute("SELECT * FROM presets WHERE id=?", (identifier,)).fetchone()
            if not row or not row["share_token_hash"] or not secrets.compare_digest(row["share_token_hash"], hashlib.sha256(token.encode()).hexdigest()):
                raise ServiceError(404, "not_found", "Convite inválido ou revogado")
            if row["owner_id"] != workspace_id:
                db.execute("INSERT OR IGNORE INTO preset_grants VALUES (?,?,?)", (identifier, workspace_id, time.time()))
        return dict(row)

    def preset_grants(self, identifier: str) -> int:
        with self.connection() as db:
            return db.execute("SELECT COUNT(*) FROM preset_grants WHERE preset_id=?", (identifier,)).fetchone()[0]

    def leave_preset(self, identifier: str, workspace_id: str) -> None:
        with self.connection() as db:
            if not db.execute("DELETE FROM preset_grants WHERE preset_id=? AND workspace_id=?", (identifier, workspace_id)).rowcount:
                raise ServiceError(404, "not_found", "Preset não encontrado")

    def delete_preset(self, identifier: str, owner_id: str) -> None:
        self.preset(identifier, owner_id)
        with self.connection() as db:
            db.execute("DELETE FROM presets WHERE id=?", (identifier,))
            db.execute("DELETE FROM preset_grants WHERE preset_id=?", (identifier,))

    def save_draft(self, owner_id: str, options: dict) -> None:
        with self.connection() as db:
            db.execute("INSERT OR REPLACE INTO generation_drafts VALUES (?,?,?)", (owner_id, json.dumps(options, ensure_ascii=False), time.time()))

    def draft(self, owner_id: str) -> dict | None:
        with self.connection() as db:
            row = db.execute("SELECT options FROM generation_drafts WHERE owner_id=?", (owner_id,)).fetchone()
        return json.loads(row[0]) if row else None

    def create_wizard_draft(self, owner_id: str, state: dict) -> str:
        identifier = uuid4().hex
        self.save_wizard_draft(identifier, owner_id, state)
        return identifier

    def save_wizard_draft(self, identifier: str, owner_id: str, state: dict) -> None:
        with self.connection() as db:
            existing = db.execute("SELECT owner_id FROM wizard_drafts WHERE id=?", (identifier,)).fetchone()
            if existing and existing["owner_id"] != owner_id:
                raise ServiceError(404, "not_found", "Rascunho não encontrado")
            db.execute("INSERT OR REPLACE INTO wizard_drafts VALUES (?,?,?,?)",
                       (identifier, owner_id, json.dumps(state, ensure_ascii=False), time.time()))

    def wizard_draft(self, identifier: str, owner_id: str) -> dict:
        with self.connection() as db:
            row = db.execute("SELECT state FROM wizard_drafts WHERE id=? AND owner_id=?", (identifier, owner_id)).fetchone()
        if not row:
            raise ServiceError(404, "not_found", "Rascunho não encontrado; inicie uma nova geração")
        return json.loads(row[0])

    def claim_wizard_draft(self, identifier: str, owner_id: str) -> dict:
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT state FROM wizard_drafts WHERE id=? AND owner_id=?", (identifier, owner_id)).fetchone()
            if not row:
                raise ServiceError(404, "not_found", "Rascunho não encontrado")
            state = json.loads(row[0])
            if state.get("submitted_job") or state.get("submitting"):
                raise ServiceError(409, "already_submitted", "Esta geração já foi enviada")
            state["submitting"] = True
            db.execute("UPDATE wizard_drafts SET state=?,updated_at=? WHERE id=?",
                       (json.dumps(state, ensure_ascii=False), time.time(), identifier))
        return state

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
