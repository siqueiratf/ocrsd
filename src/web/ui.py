import base64
import functools
import hashlib
import json
import os
import re
import tempfile
import time
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Callable
from urllib.parse import quote, urlencode
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.concurrency import run_in_threadpool
from starlette.background import BackgroundTask
from pydantic import BaseModel, ValidationError
from starlette.datastructures import FormData, UploadFile

from src.document_types import safe_document_type_key
from src.web.models import GenerationOptions, Identifier, ModelPromote, ModelUpdate, SemanticDefinitionCreate, ShareCreate, Writing
from src.web.palette import (CUSTOM_BASES, EDITOR_OPTIONS, GENERATION_RULES, ICON_NAMES, LAYOUT_PALETTE, PALETTE, TYPE_LABELS,
                             custom_field_config, custom_palette)
from src.web.services import (BUILTIN_PROTECTED, GENERATION_MODES, ORIENTATIONS, PAGE_SIZES, PREVIEW_SAMPLES, STANDARD_EDITS_DISABLED, Catalog, ServiceError,
                              create_blank_model, create_model, duplicate_model, fit_mapping, mapping_issues, preview_generation, preview_key, preview_strip,
                              promote_model, render_writing_samples, realism_previews, render_layout_pdf, render_pdf_page, pdf_page_count, render_preview,
                              revision, save_mapping, validate_layout, semantic_inferred, unavailable_custom_fields, validate_mapping, with_approved_semantics,
                              writing_issues, writing_message)
from src.web.settings import Settings
from src.web.writing import MODEL_INFO, NEURAL_CHARACTERS, adapt_to_model, model_defaults, preset_title, writing_summary

ROOT = Path(__file__).resolve().parent
WORKSPACE_COOKIE = "ocrsd_workspace"
SHARE_COOKIE = "ocrsd_share"
FILTERS = {"none": "Sem realismo", "light": "Leve", "medium": "Médio", "strong": "Forte"}
STATUS = {"queued": "Em fila", "running": "Gerando", "succeeded": "Concluído", "failed": "Falhou",
          "cancelled": "Cancelado", "expired": "Expirado"}
DNV_GROUPS = {"I": "Identificação do recém-nascido", "II": "Local da ocorrência", "III": "Dados da mãe",
              "IV": "Responsável legal", "V": "Gestação e parto", "VI": "Anomalias congênitas",
              "VII": "Preenchimento da declaração", "VIII": "Registro civil"}
ACTIVE = {"queued", "running"}
PRESET_PAGE_SIZE = 20
PRESET_TABS = {"todos": "Todos", "meus": "Meus presets", "compartilhados": "Compartilhados", "padroes": "Padrões"}
PRESET_SORTS = {"modificados": "Modificados recentemente", "recentes": "Criados recentemente", "nome": "Nome"}
MAPPING_ID = re.compile(r"^[A-Za-z0-9_-]{1,100}$")
JSON_PREVIEW_LIMIT = 2 * 1024 * 1024
LABELS = {"name": "Nome", "count": "Quantidade de documentos", "seed": "Seed aleatória", "fields": "Campos",
          "document_type": "Modelo", "mapping_id": "Mapeamento", "filter_preset": "Realismo visual",
          "model": "Modelo de escrita", "intensity": "Variação da escrita", "writer": "Identidade do escritor",
          "bias": "Regularidade", "color": "Cor", "width": "Espessura", "pressure": "Pressão", "dropout": "Falhas",
          "text_policy": "Caracteres não suportados", "label": "Nome do modelo", "description": "Descrição",
          "expires_in_hours": "Validade do link", "generation_mode": "Modo de geração", "key": "Chave do documento"}
BLOCKED_FIELD = "Sem gerador validado"
AI_FAILURE = "A geração assistida por IA falhou:"
try:
    TIMEZONE = ZoneInfo("America/Sao_Paulo")
except ZoneInfoNotFoundError:
    TIMEZONE = None


def file_response(path: Path, media_type: str, filename: str, download: bool = True) -> StreamingResponse:
    async def content():
        with path.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                yield chunk
    disposition = "attachment" if download else "inline"
    headers = {"Content-Length": str(path.stat().st_size),
               "Content-Disposition": f"{disposition}; filename*=UTF-8''{quote(filename)}"}
    return StreamingResponse(content(), media_type=media_type, headers=headers)


def format_date(timestamp: float | None) -> str:
    if not timestamp:
        return "—"
    return datetime.fromtimestamp(timestamp, TIMEZONE).strftime("%d/%m/%Y %H:%M")


def format_size(size: int) -> str:
    return f"{size / 1024 / 1024:.1f} MB" if size >= 1024 * 1024 else f"{max(1, round(size / 1024))} KB"


def format_duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "—"
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes:02d}:{seconds:02d}"


def text(form: FormData, key: str) -> str:
    value = form.get(key)
    return value.strip() if isinstance(value, str) else ""


def parsed(form: FormData, key: str, kind: type, default: Any = None) -> Any:
    value = text(form, key)
    if not value:
        return default
    try:
        return kind(value)
    except ValueError:
        return value


def validated(model: type[BaseModel], data: dict) -> Any:
    try:
        return model.model_validate(data)
    except ValidationError as exc:
        error = exc.errors()[0]
        names = [str(part) for part in error["loc"] if str(part) in LABELS]
        if names and names[-1] == "fields":
            message = "Selecione ao menos um campo existente para gerar"
        elif names:
            message = f"Confira o valor de “{LABELS[names[-1]]}”"
        else:
            message = str(error["msg"]).removeprefix("Value error, ")
        raise ServiceError(422, "validation_error", message) from exc


def writing_state(form: FormData) -> dict:
    return {"model": text(form, "model") or "font", "handwriting": "handwriting" in form,
            "text_handwriting": "text_handwriting" in form, "intensity": parsed(form, "intensity", float, 0.65),
            "writer": parsed(form, "writer", int), "bias": parsed(form, "bias", float, 0.75),
            "color": text(form, "color") or "#193787", "width": parsed(form, "width", float, 1.5),
            "pressure": parsed(form, "pressure", float, 0.25), "dropout": parsed(form, "dropout", float, 0.0),
            "text_policy": text(form, "text_policy") or "normalize"}


def options_state(form: FormData) -> dict:
    return {"preset": text(form, "preset"), "name": text(form, "name"), "document_type": text(form, "document_type"),
            "mapping_id": text(form, "mapping_id"), "fields": [str(item) for item in form.getlist("fields")],
            "count": parsed(form, "count", int, 1), "seed": parsed(form, "seed", int),
            "writing": writing_state(form), "filter_preset": text(form, "filter_preset") or "none",
            "generation_mode": text(form, "generation_mode") or "traditional"}


def set_cookie(response: Response, request: Request, name: str, value: str, max_age: int, path: str = "/") -> None:
    response.set_cookie(name, value, max_age=max_age, path=path, httponly=True, samesite="lax",
                        secure=request.url.scheme == "https")


def mount_ui(app: FastAPI, settings: Settings, store, catalog: Catalog, probe=None) -> Callable[[Request, int, str], Response]:
    templates = Jinja2Templates(directory=ROOT / "templates")
    templates.env.filters.update(data=format_date, tamanho=format_size)
    templates.env.globals.update(FILTERS=FILTERS, STATUS=STATUS, ACTIVE=ACTIVE, TYPE_LABELS=TYPE_LABELS,
                                 PAGE_SIZES=PAGE_SIZES, ORIENTATIONS=ORIENTATIONS, GENERATION_MODES=GENERATION_MODES,
                                 STANDARD_EDITS=settings.standard_edits)
    app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")

    @functools.lru_cache(maxsize=128)
    def asset_digest(name: str, modified: int) -> str:
        return hashlib.sha256((ROOT / "static" / name).read_bytes()).hexdigest()[:12]

    def asset(name: str) -> str:
        path = ROOT / "static" / name
        return f"/static/{name}?v={asset_digest(name, path.stat().st_mtime_ns)}"

    templates.env.globals["asset"] = asset
    router = APIRouter(include_in_schema=False)

    def render(request: Request, name: str, status_code: int = 200, headers: dict | None = None, **context) -> HTMLResponse:
        response = templates.TemplateResponse(request, name, context, status_code=status_code, headers=headers)
        token = getattr(request.state, "workspace_token", None)
        if token:
            set_cookie(response, request, WORKSPACE_COOKIE, token, 400 * 86400)
        return response

    def interface_error(request: Request, status: int, message: str) -> Response:
        if status == 404 and message == "Recurso ou operação indisponível":
            message = "Página ou recurso não encontrado"
        if request.headers.get("hx-request") == "true":
            return render(request, "partials/toast.html", status, {"HX-Retarget": "#avisos", "HX-Reswap": "beforeend"},
                          message=message, kind="erro")
        return render(request, "error.html", status, code=status, message=message)

    def owner(request: Request) -> str:
        token = request.cookies.get(WORKSPACE_COOKIE)
        if not token:
            raise ServiceError(401, "workspace_required", "Sessão ausente; recarregue a página")
        return store.workspace(token)

    def page_owner(request: Request) -> str:
        token = request.cookies.get(WORKSPACE_COOKIE)
        if token:
            try:
                return store.workspace(token)
            except ServiceError:
                pass
        identifier, token = store.create_workspace()
        request.state.workspace_token = token
        return identifier

    Owner = Annotated[str, Depends(owner)]
    PageOwner = Annotated[str, Depends(page_owner)]

    def toast(request: Request, message: str) -> HTMLResponse:
        return render(request, "partials/toast.html", message=message, kind="sucesso", oob=True)

    def model_status(model: str) -> tuple[bool, str | None]:
        found = next((item for item in catalog.models() if item["id"] == model), None)
        return (bool(found and found["available"]), None if found and found["available"] else (found or {}).get("reason") or "Modelo desconhecido")

    def preset_entry(row: dict, kind: str) -> dict:
        writing = json.loads(row["writing"])
        available, reason = model_status(writing["model"])
        return {"id": row["id"], "name": row["name"], "title": row["name"], "description": row.get("description") or "", "writing": writing,
                "filter_preset": row["filter_preset"], "available": available, "reason": reason, "kind": kind, "editable": kind == "meu",
                "created_at": row["created_at"], "updated_at": row["updated_at"], "version": row.get("version") or 1,
                "shared": bool(row.get("share_token_hash")), "summary": writing_summary(writing)}

    def presets(owner_id: str) -> list[dict]:
        items = []
        for item in catalog.presets():
            writing = item.writing.model_dump() if item.writing else None
            items.append({"id": item.id, "name": item.label, "title": preset_title(item.label, writing), "description": "",
                          "writing": writing, "filter_preset": item.filter_preset, "available": item.available, "reason": item.reason,
                          "kind": "padrao", "editable": False, "created_at": None, "updated_at": None, "version": 1, "shared": False,
                          "summary": writing_summary(writing)})
        items += [preset_entry(row, "meu") for row in store.list_presets(owner_id)]
        items += [preset_entry(row, "compartilhado") for row in store.list_shared_presets(owner_id)]
        return items

    def find_preset(owner_id: str, identifier: str) -> dict:
        found = next((item for item in presets(owner_id) if item["id"] == identifier), None)
        if not found:
            raise ServiceError(404, "not_found", "Preset não encontrado")
        return found

    def preset(owner_id: str, identifier: str) -> dict:
        found = find_preset(owner_id, identifier)
        if not found["writing"]:
            raise ServiceError(422, "preset_mixed", found["reason"] or "Preset disponível somente no desktop")
        if not found["available"]:
            raise ServiceError(422, "model_unavailable", found["reason"] or "Preset indisponível")
        return found

    def preset_thumbnail(item: dict, rows: int = 4) -> str | None:
        if not item["writing"] or not item["available"]:
            return None
        return f'/presets/escrita/{item["id"]}/amostra.png?linhas={rows}&v={preview_key(Writing.model_validate(item["writing"]), rows)}'

    def strip_presets(owner_id: str, selected: str) -> list[dict]:
        usable = [item for item in presets(owner_id) if item["writing"] and item["available"]]
        ordered = sorted(usable, key=lambda item: (item["id"] != selected, item["kind"] == "padrao", -(item["updated_at"] or 0)))
        return [{**item, "thumbnail": preset_thumbnail(item, 1)} for item in ordered[:8]]

    def model_entry(owner_id: str, model_id: str):
        model = next((item for item in catalog.document_models(store.list_models(owner_id)) if item.id == model_id), None)
        if model is None:
            raise ServiceError(404, "not_found", "Modelo não encontrado")
        return model

    def wizard_initial() -> dict:
        return {"document_type": "", "mapping_id": "", "fields": [], "count": 10 if settings.max_count is None else min(10, settings.max_count),
                "preset": "padrao_fonte_natural", "filter_preset": "none", "generation_mode": "traditional", "step": 1}

    def assistant_summary() -> dict:
        if probe is None:
            return {"available": False, "provider": settings.ai_provider, "model": settings.ai_model, "detail": "Assistente de IA desativado",
                    "hint": ""}
        return probe.summary()

    def reused_state(owner_id: str, row: dict) -> dict:
        options = store.detail(row).options
        if options is None:
            raise ServiceError(422, "invalid_configuration", "A configuração original não está disponível")
        reference = json.loads(row["preset"]) if row.get("preset") else None
        state = {**wizard_initial(), "document_type": options.document_type, "mapping_id": options.mapping_id,
                 "fields": list(options.fields), "count": options.count, "filter_preset": options.filter_preset,
                 "generation_mode": options.generation_mode,
                 "preset": "original_snapshot", "writing_snapshot": options.writing.model_dump(),
                 "original_preset_name": reference["name"] if reference else "Configuração original", "step": 6}
        try:
            state["mapping_id"] = catalog.active_mapping(options.document_type, owner_id)
            definition = catalog.definition(options.document_type, state["mapping_id"], owner_id)
            available = {str(field["field_id"]) for field in definition["fields"]}
            if set(options.fields) - available:
                state["original_fields_missing"] = True
                state["fields"] = [item for item in options.fields if item in available]
                state["step"] = 2
        except ServiceError:
            state["original_mapping_missing"] = True
            state["step"] = 1
        if not model_status(options.writing.model)[0]:
            state["original_model_unavailable"] = True
        return state

    def wizard_context(owner_id: str, state: dict, identifier: str, step: int, error: str = "", assistant: dict | None = None) -> dict:
        owned = store.list_models(owner_id)
        published = [row for row in owned if row["status"] == "published"]
        unpublished = next((row for row in owned if row["id"] == state.get("document_type") and row["status"] != "published"), None)
        types = catalog.document_types(published)
        models = {item.id: item for item in catalog.document_models(published)}
        current = next((item for item in types if item["id"] == state.get("document_type")), None)
        if current and state.get("mapping_id") != current["active_mapping"]:
            state["mapping_id"] = current["active_mapping"]
        mapping = next((item for item in current["mappings"] if item["id"] == current["active_mapping"]), None) if current else None
        definition = catalog.definition(current["id"], mapping["id"], owner_id) if mapping else None
        dnv_positions = []
        if current and current["id"] == "dnv" and definition:
            try:
                reference = catalog.definition("dnv", "mapeamento_seguindo_norma", owner_id)
                dnv_positions = [((item["bbox"][1] + item["bbox"][3]) / (2 * reference["page_size_px"][1]), item["bloco"])
                                 for item in reference["fields"] if item.get("bloco") and item.get("bbox")]
            except ServiceError:
                pass
        fields = []
        blocked = unavailable_custom_fields(definition) if definition else {}
        if definition:
            for field in definition["fields"]:
                section = field.get("bloco")
                if not section and dnv_positions and field.get("bbox"):
                    midpoint = (field["bbox"][1] + field["bbox"][3]) / (2 * definition["page_size_px"][1])
                    section = min(dnv_positions, key=lambda item: abs(item[0] - midpoint))[1]
                fields.append({"id": str(field["field_id"]), "label": field.get("label") or field.get("description") or str(field["field_id"]),
                               "data_type": field.get("data_type", "text"),
                               "group": field.get("section_label") or field.get("group_label") or (DNV_GROUPS.get(str(section)) if current and current["id"] == "dnv" else None) or section or "Outros campos",
                               "description": field.get("description") or "", "official_variable": field.get("official_variable") or "",
                               "bbox": field.get("bbox"), "elements": field.get("elements") or [],
                               "blocked": field.get("unsupported_reason") or (BLOCKED_FIELD if str(field["field_id"]) in blocked else "")})
        groups = {}
        for field in fields:
            groups.setdefault(field["group"], []).append(field)
        if "Outros campos" in groups and len(groups) > 1:
            groups["Outros campos"] = groups.pop("Outros campos")
        styles = [{**item, "thumbnail": preset_thumbnail(item, 1)} for item in presets(owner_id) if item["writing"] and item["available"]]
        styles.sort(key=lambda item: (item["id"] != "padrao_fonte_natural", item["id"] != state.get("preset"), item["kind"] != "meu", item["title"]))
        if state.get("writing_snapshot"):
            styles.insert(0, {"id": "original_snapshot", "title": state.get("original_preset_name") or "Configuração original",
                              "summary": writing_summary(state["writing_snapshot"]), "thumbnail": f"/geracao/{identifier}/amostra.png",
                              "writing": state["writing_snapshot"]})
        notices = []
        if state.get("original_mapping_missing") and not mapping:
            notices.append("O modelo original não está mais disponível. Selecione outro documento antes de gerar.")
        if state.get("original_fields_missing"):
            notices.append("Alguns campos da geração original não existem no mapeamento atual do modelo. Revise a seleção de campos.")
        if state.get("original_model_unavailable") and state.get("preset") == "original_snapshot":
            notices.append("O modelo de escrita original não está disponível. Selecione outro estilo.")
        if step == 4 and definition:
            for item in styles:
                writing = Writing.model_validate(item["writing"])
                conflicts = writing_issues(definition, state.get("fields", []), writing)
                item["conflict"] = writing_message(writing, conflicts) if conflicts else ""
            if any(item["id"] == state.get("preset") and item["conflict"] for item in styles):
                notices.append("O estilo de escrita escolhido não é compatível com os campos selecionados. Escolha outro estilo.")
        selected_style = next((item for item in styles if item["id"] == state.get("preset")), None)
        if step == 6 and not state.get("submitted_job") and not error:
            try:
                if not selected_style:
                    raise ServiceError(422, "preset_unavailable", "O preset de escrita selecionado não está disponível. Volte à etapa Escrita.")
                data = {"name": f"Dataset {state['document_type']}", "document_type": state["document_type"],
                        "mapping_id": state["mapping_id"], "fields": state["fields"], "count": state["count"],
                        "writing": selected_style["writing"], "filter_preset": state["filter_preset"],
                        "generation_mode": state.get("generation_mode", "traditional")}
                catalog.validate(validated(GenerationOptions, data), owner_id)
            except ServiceError as exc:
                error = exc.message
        return {"draft_id": identifier, "state": state, "wizard_step": step, "wizard_error": error,
                "wizard_notice": " ".join(notices),
                "types": types, "model_info": models, "current_type": current, "mapping": mapping,
                "definition": definition, "fields": fields, "groups": groups, "styles": styles,
                "selected_style": selected_style, "assistant": assistant, "selectable": [field for field in fields if not field["blocked"]],
                "max_count": settings.max_count, "unpublished": unpublished}

    def wizard_url(identifier: str, step: int) -> str:
        return f"/geracao?draft={identifier}&etapa={step}"

    def wizard_redirect(request: Request, identifier: str, step: int) -> Response:
        url = wizard_url(identifier, step)
        return Response(headers={"HX-Redirect": url}) if request.headers.get("hx-request") == "true" else RedirectResponse(url, 303)

    def generation_context(owner_id: str, options: dict, search: str = "") -> dict:
        drafts = {row["id"]: row["label"] for row in store.list_models(owner_id) if row["status"] != "published"}
        types = catalog.document_types([row for row in store.list_models(owner_id) if row["id"] not in drafts])
        draft = {"id": options["document_type"], "label": drafts[options["document_type"]]} if options["document_type"] in drafts else None
        current = next((item for item in types if item["id"] == options["document_type"]), types[0])
        mappings = [item["id"] for item in current["mappings"]]
        if options["document_type"] != current["id"] or options["mapping_id"] not in mappings:
            options.update(document_type=current["id"], mapping_id=mappings[0], fields=None)
        definition = catalog.definition(current["id"], options["mapping_id"], owner_id)
        fields = [{"id": str(field["field_id"]), "label": field.get("label") or field.get("description") or str(field["field_id"]),
                   "data_type": field.get("data_type", "text")} for field in definition["fields"]]
        available = [field["id"] for field in fields]
        options["fields"] = available if options["fields"] is None else [item for item in options["fields"] if item in available]
        needle = search.casefold()
        for field in fields:
            field["hidden"] = bool(needle) and needle not in f'{field["label"]} {field["id"]}'.casefold()
        chosen = next((item for item in presets(owner_id) if item["id"] == options.get("preset") and item["writing"]), None)
        options["preset"] = chosen["id"] if chosen else ""
        adjusted = bool(chosen) and chosen["writing"] != options["writing"]
        return {"options": options, "types": types, "current_type": current, "fields": fields, "search": search, "draft": draft,
                "strip": strip_presets(owner_id, options["preset"]), "chosen_preset": chosen, "preset_adjusted": adjusted,
                "models": catalog.models(), "presets": presets(owner_id), "configs": store.list_configs(owner_id),
                "max_count": settings.max_count}

    def owned_job(owner_id: str, job_id: str) -> dict:
        return store.authorize_owner(store.get(job_id), owner_id)

    def safe_job_error(message: str | None) -> str:
        known = {"A geração excedeu o limite de memória do worker", "Um arquivo necessário ao modelo de geração não foi encontrado",
                 "A geração foi interrompida por uma falha interna", "Tempo máximo de execução excedido",
                 "Limite de armazenamento do job excedido", "Espaço livre insuficiente"}
        if message in known:
            return message
        if message and message.startswith(AI_FAILURE):
            return message
        if message and message.startswith("A geração foi rejeitada pelo motor:"):
            return "Os parâmetros foram rejeitados pelo motor de geração"
        return "A geração foi interrompida por uma falha interna"

    def result_context(owner_id: str, job_id: str, document_number: int = 1) -> dict:
        row = owned_job(owner_id, job_id)
        artifacts, problem = [], None
        if row["status"] in {"succeeded", "running", "failed", "cancelled"}:
            try:
                artifacts = store.artifacts(row)
            except ServiceError as exc:
                problem = exc.message
        by_name = {item["name"]: item for item in artifacts}
        documents = []
        generation = None
        manifest_artifact = by_name.get("dataset.json")
        if manifest_artifact:
            path, _ = store.artifact(row, manifest_artifact["id"])
            try:
                manifest = json.loads(path.read_text(encoding="utf-8"))
                generation = manifest.get("generation") if isinstance(manifest.get("generation"), dict) else None
                for entry in manifest.get("documents", []):
                    if not isinstance(entry, dict):
                        continue
                    pdf = by_name.get(entry.get("pdf"))
                    ground_truth = by_name.get(entry.get("ground_truth"))
                    if pdf and pdf["media_type"] == "application/pdf":
                        number = len(documents) + 1
                        documents.append({"id": str(number), "number": number, "label": f"Documento {number:03d}", "pdf": pdf,
                                          "json": ground_truth if ground_truth and ground_truth["media_type"] == "application/json" else None})
            except (ValueError, OSError, TypeError):
                problem = "O manifesto do dataset não pôde ser lido"
        elif artifacts:
            for number in sorted({item.get("document_number") for item in artifacts if item.get("document_number")}):
                pdf = next((item for item in artifacts if item.get("document_number") == number and item["media_type"] == "application/pdf"), None)
                ground_truth = next((item for item in artifacts if item.get("document_number") == number and item["media_type"] == "application/json"), None)
                if pdf and ground_truth:
                    documents.append({"id": str(number), "number": number, "label": f"Documento {number:03d}",
                                      "pdf": pdf, "json": ground_truth})
        selected = next((item for item in documents if item["number"] == document_number), None)
        if documents and selected is None:
            raise ServiceError(404, "not_found", "Documento não encontrado")
        pages = 0
        if selected:
            path, _ = store.artifact(row, selected["pdf"]["id"])
            try:
                pages = pdf_page_count(path)
            except Exception:
                problem = "O PDF selecionado não pôde ser aberto"
        labels = {item["id"]: item["label"] for item in catalog.document_types(store.list_models(owner_id))}
        detail = store.detail(row)
        return {"job": detail, "documents": documents, "selected_document": selected, "pages": pages,
                "document_label": labels.get(detail.options.document_type, detail.options.document_type) if detail.options else "Documento",
                "safe_error": safe_job_error(row.get("error")),
                "problem": problem, "preset": json.loads(row["preset"]) if row.get("preset") else None,
                "archive": next((item for item in artifacts if item["media_type"] == "application/zip"), None),
                "manifest": manifest_artifact, "generation": generation_info(row, detail, generation)}

    def generation_info(row: dict, detail, generation: dict | None) -> dict:
        mode = (generation or {}).get("mode") or ("ai_replay" if row.get("replay_of") else
                                                  detail.options.generation_mode if detail.options else "traditional")
        mode = mode if mode in GENERATION_MODES else "traditional"
        info = {"mode": mode, "label": GENERATION_MODES[mode], "source_job": row.get("replay_of") or (generation or {}).get("source_job")}
        if mode == "traditional":
            return info
        records = store.records(row) or {}
        parameters = (generation or {}).get("source_generation") if mode == "ai_replay" else generation
        parameters = parameters or (records.get("generation") if mode == "ai_replay" else None) or {}
        documents = (generation or {}).get("documents") if mode == "ai" else (records.get("generation") or {}).get("documents")
        documents = [item for item in documents or [] if isinstance(item, dict)]
        digest = str(parameters.get("model_digest") or "").removeprefix("sha256:")
        info["ai"] = {"provider": parameters.get("provider"), "model": parameters.get("model"), "digest": digest[:12] or None,
                      "prompt_version": parameters.get("prompt_version"), "temperature": parameters.get("temperature"),
                      "accepted": sum(item.get("status") == "accepted" for item in documents) if documents else None,
                      "attempts": sum(int(item.get("attempts") or 0) for item in documents) if documents else None} if parameters else None
        return info

    def jobs_context(owner_id: str, search: str, status: str, selected: str = "", document_type: str = "", order: str = "recentes", page: int = 1) -> dict:
        status = ",".join(item for item in dict.fromkeys(status.split(",")) if item in STATUS)
        order = order if order in {"recentes", "antigos", "data_asc", "data_desc", "nome_asc", "nome_desc", "modelo_asc", "modelo_desc", "quantidade_asc", "quantidade_desc", "tempo_asc", "tempo_desc"} else "recentes"
        page = max(1, page)
        jobs, total = store.list_jobs_page(owner_id, search[:120], status, document_type[:100], order, page)
        labels = {item["id"]: item["label"] for item in catalog.document_types(store.list_models(owner_id))}
        for job in jobs:
            model_id = job["options"]["document_type"] if job["options"] else ""
            job["document_label"] = labels.get(model_id, model_id or "—")
            end = job["finished_at"] if job["finished_at"] else (time.time() if job["status"] == "running" else None)
            job["elapsed"] = round(end - job["started_at"], 1) if end and job["started_at"] else None
            job["duration"] = format_duration(job["elapsed"])
        filters = {"busca": search[:120], "status": status, "modelo": document_type[:100], "ordem": order, "pagina": page}
        detail_query = urlencode({key: value for key, value in filters.items() if value and value not in {"recentes", "data_desc", 1}})
        return {"jobs": jobs, "search": search[:120], "status": status, "selected": selected, "document_type": document_type,
                "order": order, "page_number": page, "page_count": max(1, (total + 24) // 25), "total": total,
                "detail_query": detail_query,
                "selected_statuses": status.split(",") if status else [],
                "document_types": store.job_document_types(owner_id),
                "polling": any(job["status"] in ACTIVE for job in jobs)}

    @router.get("/")
    async def index():
        return RedirectResponse("/modelos", 303)

    @router.get("/favicon.ico")
    async def favicon():
        return FileResponse(ROOT / "static" / "favicon.ico", media_type="image/x-icon", headers={"Cache-Control": "public, max-age=86400"})

    @router.get("/apple-touch-icon.png")
    async def touch_icon():
        return FileResponse(ROOT / "static" / "apple-touch-icon.png", media_type="image/png", headers={"Cache-Control": "public, max-age=86400"})

    @router.post("/sessao")
    async def adopt_session(request: Request):
        form = await request.form()
        token = text(form, "token")
        store.workspace(token)
        response = Response(status_code=204, headers={"HX-Refresh": "true"})
        set_cookie(response, request, WORKSPACE_COOKIE, token, 400 * 86400)
        return response

    def gallery(owner_id: str) -> list[dict]:
        items = []
        for model in catalog.document_models(store.list_models(owner_id)):
            mappings = [path.stem for path in catalog.types[model.id].mappings] if model.builtin else [model.mapping_id]
            items.append({"model": model, "mappings": mappings, "composed": not model.builtin and store.layout(model.id) is not None})
        return items

    def editor_paths(owner_id: str, model) -> tuple[Path, ...]:
        mapping = catalog.mapping(model.id, model.mapping_id, None if model.builtin else owner_id)
        return (mapping,) if model.builtin else (mapping, store.model_directory(model.id) / "layout.json")

    def editor_state(owner_id: str, model) -> dict:
        definition = catalog.definition(model.id, model.mapping_id, None if model.builtin else owner_id)
        layout = None if model.builtin else store.layout(model.id)
        mapping = editor_paths(owner_id, model)[0]
        return {"definition": definition, "layout": layout, "revision": revision(*editor_paths(owner_id, model)),
                "mapping_revision": hashlib.sha256(mapping.read_bytes()).hexdigest()}

    def editor_view(owner_id: str, model) -> dict:
        state = editor_state(owner_id, model)
        custom = [{"key": item["key"], "label": item["label"], "description": item["description"],
                   "group": "Personalizados", "kind": item["data_type"]}
                  for item in store.list_custom_semantics(owner_id)]
        latest = None if model.builtin else store.latest_analysis(model.id, owner_id)
        return {**state, "semantic_inferred": semantic_inferred(state["definition"], state["definition"].get("template_id") or model.id),
                "semantic_options": [*EDITOR_OPTIONS["semantic_options"], *custom],
                "analysis": latest and {name: latest[name] for name in ("id", "status", "progress", "stage", "error")},
                "model": {"id": model.id, "label": model.label, "builtin": model.builtin, "status": model.status,
                          "version": model.version, "editable": model.editable}}

    def assist_model(owner_id: str, model_id: str):
        model = model_entry(owner_id, model_id)
        if model.builtin:
            raise ServiceError(403, "builtin_protected", BUILTIN_PROTECTED)
        if store.layout(model.id) is not None:
            raise ServiceError(422, "mapping_assist_unavailable",
                               "Documentos compostos não usam o mapeamento com IA; mapeie os campos manualmente")
        if model.status != "draft":
            raise ServiceError(422, "mapping_assist_unavailable", "Volte o modelo para rascunho antes de analisar o PDF")
        return model

    def publication(request: Request, model, issues: list[str], message: str = "", oob: bool = False) -> HTMLResponse:
        return render(request, "partials/publication.html", model=model, issues=issues, message=message, oob=oob)

    @router.get("/modelos")
    async def models_page(request: Request, owner_id: PageOwner):
        return render(request, "models.html", page="modelos", models=gallery(owner_id))

    @router.get("/modelos/novo")
    async def new_model(request: Request, owner_id: PageOwner, modo: str = ""):
        return render(request, "model_new.html", page="modelos", mode=modo if modo in {"importar", "branco"} else "",
                      max_upload=settings.max_upload_bytes)

    @router.post("/modelos")
    async def create(request: Request, owner_id: Owner):
        async with request.form() as form:
            return await create_from_form(form, owner_id)

    async def create_from_form(form: FormData, owner_id: str) -> Response:
        pdf, mapping_file = form.get("pdf"), form.get("mapping")
        if not isinstance(pdf, UploadFile) or not pdf.filename:
            raise ServiceError(422, "invalid_pdf", "Selecione um template PDF")
        metadata = validated(ModelUpdate, {"label": text(form, "label"), "description": text(form, "description")})
        mapping_id = text(form, "mapping_id") or "default"
        if not MAPPING_ID.match(mapping_id):
            raise ServiceError(422, "validation_error", "Use somente letras, números, “_” e “-” no ID do mapeamento")
        mapping = None
        if isinstance(mapping_file, UploadFile) and mapping_file.filename:
            try:
                mapping = json.loads(await mapping_file.read())
            except ValueError as exc:
                raise ServiceError(422, "invalid_mapping", "O mapeamento enviado não é um JSON válido") from exc
            if not isinstance(mapping, dict):
                raise ServiceError(422, "invalid_mapping", "O mapeamento precisa ser um objeto JSON")
        row = create_model(store, settings, owner_id, metadata.label, metadata.description, mapping_id,
                           pdf.filename, await pdf.read(), mapping)
        return Response(headers={"HX-Redirect": f'/modelos/{row["id"]}/editor'})

    @router.post("/modelos/em-branco")
    async def create_blank(request: Request, owner_id: Owner):
        form = await request.form()
        metadata = validated(ModelUpdate, {"label": text(form, "label"), "description": text(form, "description")})
        row = create_blank_model(store, owner_id, metadata.label, metadata.description, text(form, "size"), text(form, "orientation"))
        return Response(headers={"HX-Redirect": f'/modelos/{row["id"]}/editor'})

    @router.post("/modelos/{model_id}/copia")
    async def duplicate(model_id: Identifier, owner_id: Owner):
        row = duplicate_model(catalog, store, model_id, owner_id)
        return Response(headers={"HX-Redirect": f'/modelos/{row["id"]}/editor'})

    @router.get("/modelos/{model_id}/salvar-padrao")
    async def promote_dialog(request: Request, model_id: Identifier, owner_id: Owner):
        if not settings.standard_edits:
            raise ServiceError(403, "standard_edits_disabled", STANDARD_EDITS_DISABLED)
        model = model_entry(owner_id, model_id)
        if model.builtin:
            raise ServiceError(404, "not_found", "Modelo não encontrado")
        return render(request, "partials/promote_dialog.html", model_id=model.id,
                      key=safe_document_type_key(model.label), label=model.label, description=model.description)

    @router.post("/modelos/{model_id}/salvar-padrao")
    async def promote(request: Request, model_id: Identifier, owner_id: Owner):
        form = await request.form()
        payload = validated(ModelPromote, {"key": text(form, "key"), "label": text(form, "label"),
                                           "description": text(form, "description")})
        promote_model(catalog, store, model_id, owner_id, payload.key, payload.label, payload.description)
        return Response(headers={"HX-Redirect": "/modelos"})

    @router.get("/modelos/{model_id}/previa.png")
    async def model_preview(model_id: Identifier, owner_id: Owner, largura: int = 1200):
        return Response(render_preview(catalog.document(model_id, owner_id).template, max(400, min(largura, 2600))),
                        media_type="image/png", headers={"Cache-Control": "private, max-age=300"})

    @router.get("/modelos/{model_id}/editor")
    async def editor_page(request: Request, model_id: Identifier, owner_id: PageOwner, draft: str = ""):
        model = model_entry(owner_id, model_id)
        if model.builtin and (not model.editable or not model.mapping_id):
            return RedirectResponse("/modelos", 303)
        state = editor_view(owner_id, model)
        groups = [*PALETTE, custom_palette(store.list_field_types(owner_id)), *([LAYOUT_PALETTE] if state["layout"] else [])]
        composed = state["layout"] is not None
        if draft:
            store.wizard_draft(draft, owner_id)
        return render(request, "editor.html", page="modelos", layout_page="editor", model=model, groups=groups, return_draft=draft,
                      composed=composed, issues=mapping_issues(state["definition"]), icon_names=ICON_NAMES,
                      editor={**EDITOR_OPTIONS, **state,
                              "mapping_id": model.mapping_id, "palette": groups,
                              "preview": None if composed else f'/modelos/{model.id}/previa.png?largura=2400&v={state["revision"]}',
                              "source": f"/modelos/{model.id}/estado.json"})

    @router.post("/modelos/{model_id}/mapear-ia", status_code=202)
    async def start_mapping_assist(request: Request, model_id: Identifier, owner_id: Owner):
        model = assist_model(owner_id, model_id)
        try:
            payload = await request.json()
        except ValueError as exc:
            raise ServiceError(422, "invalid_page", "Informe a página do documento") from exc
        if not isinstance(payload, dict) or type(payload.get("page")) is not int or payload["page"] != 1:
            raise ServiceError(422, "invalid_page", "O editor atual analisa a página 1 do documento")
        return JSONResponse(store.submit_analysis(model.id, owner_id, 1, request.client.host if request.client else "unknown"),
                            status_code=202)

    @router.get("/modelos/{model_id}/mapear-ia/{job_id}")
    async def mapping_assist_result(model_id: Identifier, job_id: Identifier, owner_id: Owner):
        model = model_entry(owner_id, model_id)
        if model.builtin:
            raise ServiceError(403, "builtin_protected", BUILTIN_PROTECTED)
        result = store.analysis(job_id, owner_id)
        if result["model_id"] != model.id:
            raise ServiceError(404, "not_found", "Análise não encontrada para este modelo")
        proposal = result["proposal"]
        if proposal:
            current = catalog.definition(model.id, model.mapping_id, owner_id)
            template = hashlib.sha256((store.model_directory(model.id) / "template.pdf").read_bytes()).hexdigest()
            result["proposal"] = fit_mapping(proposal, tuple(current["page_size_px"]))
            result["stale"] = (proposal.get("provenance") or {}).get("pdf_sha256") not in (None, template)
        return JSONResponse(result)

    @router.post("/modelos/{model_id}/definicoes-semanticas", status_code=201)
    async def create_semantic_definition_ui(request: Request, model_id: Identifier, owner_id: Owner):
        assist_model(owner_id, model_id)
        try:
            payload = await request.json()
        except ValueError as exc:
            raise ServiceError(422, "invalid_semantic", "Definição semântica inválida") from exc
        definition = validated(SemanticDefinitionCreate, payload)
        return JSONResponse(store.save_custom_semantic(owner_id, **definition.model_dump()), status_code=201)

    @router.get("/modelos/{model_id}/estado.json")
    async def editor_json(model_id: Identifier, owner_id: Owner):
        return JSONResponse(editor_view(owner_id, model_entry(owner_id, model_id)))

    @router.get("/modelos/{model_id}/mapeamento.json")
    async def mapping_json(model_id: Identifier, owner_id: Owner):
        model = model_entry(owner_id, model_id)
        return JSONResponse(catalog.definition(model.id, model.mapping_id, None if model.builtin else owner_id))

    def submitted_state(form: FormData, model, owner_id: str) -> tuple[dict, dict | None]:
        try:
            definition = json.loads(text(form, "definition"))
            layout = json.loads(text(form, "layout") or "null")
        except ValueError as exc:
            raise ServiceError(422, "invalid_mapping", "O mapeamento enviado não é um JSON válido") from exc
        current = editor_state(owner_id, model)
        page_size = tuple(current["definition"]["page_size_px"])
        if not isinstance(definition, dict):
            raise ServiceError(422, "invalid_mapping", "O mapeamento precisa ser um objeto JSON")
        definition = validate_mapping(with_approved_semantics(store, owner_id,
                                      {**definition, "dpi": current["definition"]["dpi"], "page_size_px": list(page_size)}), page_size)
        if (layout is None) != (current["layout"] is None):
            raise ServiceError(422, "invalid_layout", "Somente documentos compostos aceitam elementos de layout")
        return definition, layout if layout is None else validate_layout(layout, page_size)

    @router.post("/modelos/{model_id}/mapeamento")
    async def save(request: Request, model_id: Identifier, owner_id: Owner):
        form = await request.form()
        model = model_entry(owner_id, model_id)
        if model.builtin and not settings.standard_edits:
            raise ServiceError(403, "builtin_protected", BUILTIN_PROTECTED)
        if text(form, "revision") and text(form, "revision") != revision(*editor_paths(owner_id, model)):
            raise ServiceError(409, "revision_conflict", "O mapeamento foi alterado em outra janela. Recarregue a versão salva antes de continuar.")
        definition, layout = submitted_state(form, model, owner_id)
        metadata = None
        if not model.builtin:
            metadata = validated(ModelUpdate, {"label": text(form, "label") or model.label,
                                               "description": text(form, "description"), "status": model.status})
        save_mapping(catalog, store, model_id, owner_id, definition)
        if layout is not None:
            store.save_layout(model_id, owner_id, layout, render_layout_pdf(layout, tuple(definition["page_size_px"]), definition["dpi"]))
        issues = mapping_issues(definition)
        message = "Mapeamento salvo"
        if metadata:
            status = "draft" if issues and model.status == "published" else model.status
            store.update_model(model_id, owner_id, metadata.label, metadata.description, status)
            if status != model.status:
                message = "Mapeamento salvo; o modelo voltou para rascunho porque há pendências"
        model = model_entry(owner_id, model_id)
        new_revision = revision(*editor_paths(owner_id, model))
        response = publication(request, model, issues, message, oob=True)
        response.headers["HX-Trigger"] = json.dumps({"editor-salvo": {"revision": new_revision}})
        return response

    @router.post("/modelos/{model_id}/publicar")
    async def publish(request: Request, model_id: Identifier, owner_id: Owner):
        model = model_entry(owner_id, model_id)
        if model.builtin:
            raise ServiceError(409, "builtin_model", "Modelos padrão já estão disponíveis para geração")
        issues = mapping_issues(editor_state(owner_id, model)["definition"])
        if issues:
            raise ServiceError(422, "publication_blocked", "Resolva antes de publicar: " + "; ".join(issues[:3]))
        store.set_status(model_id, owner_id, "published")
        return publication(request, model_entry(owner_id, model_id), [], "Modelo publicado e disponível para geração", oob=True)

    @router.post("/modelos/{model_id}/rascunho")
    async def unpublish(request: Request, model_id: Identifier, owner_id: Owner):
        model = model_entry(owner_id, model_id)
        if model.builtin:
            raise ServiceError(409, "builtin_model", "Modelos padrão não podem voltar para rascunho")
        store.set_status(model_id, owner_id, "draft")
        model = model_entry(owner_id, model_id)
        return publication(request, model, mapping_issues(editor_state(owner_id, model)["definition"]),
                           "Modelo retirado da geração", oob=True)

    @router.post("/modelos/{model_id}/previa")
    async def preview(request: Request, model_id: Identifier, owner_id: Owner):
        form = await request.form()
        model = model_entry(owner_id, model_id)
        definition, layout = submitted_state(form, model, owner_id)
        template = catalog.document(model.id, None if model.builtin else owner_id).template
        seed = parsed(form, "seed", int, 42)
        seed = seed if isinstance(seed, int) and 0 <= seed < 2**32 else 42
        if layout is None:
            image = preview_generation(template, definition, seed)
        else:
            with tempfile.TemporaryDirectory(prefix="ocrsd-layout-") as directory:
                composed = Path(directory) / "template.pdf"
                composed.write_bytes(render_layout_pdf(layout, tuple(definition["page_size_px"]), definition["dpi"]))
                image = preview_generation(composed, definition, seed)
        return render(request, "partials/preview_dialog.html", model=model, image=image, seed=seed,
                      fields=len(definition["fields"]), issues=mapping_issues(definition))

    @router.get("/modelos/tipos/novo")
    async def field_type_dialog(request: Request, owner_id: Owner, base_type: str = "text", nome: str = "",
                                validacao: str = "", aplicar: str = ""):
        try:
            rules = json.loads(validacao) if validacao else {}
        except ValueError:
            rules = {}
        rules = rules if isinstance(rules, dict) else {}
        rule = ("lista" if rules.get("allowed_values") else "mascara" if rules.get("mask")
                else "intervalo" if "min_value" in rules or "max_value" in rules else "padrao")
        return render(request, "partials/field_type_dialog.html", bases=CUSTOM_BASES, rules=GENERATION_RULES,
                      base=base_type if base_type in CUSTOM_BASES else "text", type_name=nome[:60], rule=rule,
                      validation=rules, apply=bool(aplicar), date_formats=EDITOR_OPTIONS["date_formats"],
                      document_kinds=EDITOR_OPTIONS["document_kinds"])

    @router.post("/modelos/tipos")
    async def field_type_create(request: Request, owner_id: Owner):
        form = await request.form()
        result = custom_field_config(form)
        config = {**result["config"], "label": result["name"]}
        events = {"fechar-modal": True, "campo-personalizado": {"config": config, "title": result["name"],
                                                                "apply": text(form, "aplicar") == "1"}}
        message = "Configuração aplicada"
        if text(form, "escopo") == "tipo":
            store.save_field_type(owner_id, result["slug"], result["name"], result["description"], result["base_type"], result["config"])
            events["tipos-atualizados"] = True
            message = "Tipo personalizado salvo na paleta"
        return render(request, "partials/toast.html", headers={"HX-Trigger-After-Swap": json.dumps(events)},
                      message=message, kind="sucesso", oob=True)

    @router.get("/modelos/tipos/paleta")
    async def field_type_palette(request: Request, owner_id: Owner):
        return render(request, "partials/palette_group.html", group=custom_palette(store.list_field_types(owner_id)))

    @router.get("/presets/tipos")
    async def field_types_page(request: Request, owner_id: PageOwner):
        return render(request, "field_types.html", page="presets", field_types=store.list_field_types(owner_id))

    @router.post("/presets/tipos/{type_id}/excluir")
    async def field_type_delete(request: Request, type_id: Identifier, owner_id: Owner):
        store.delete_field_type(type_id, owner_id)
        return render(request, "partials/field_type_list.html", field_types=store.list_field_types(owner_id),
                      message="Tipo personalizado excluído")

    @router.post("/modelos/{model_id}/excluir")
    async def delete(model_id: Identifier, owner_id: Owner):
        store.delete_model(model_id, owner_id)
        return Response(headers={"HX-Redirect": "/modelos"})

    @router.get("/geracao")
    async def generation_page(request: Request, owner_id: PageOwner, modelo: str = "", job: str = "", configuracao: str = "",
                              preset: str = "", rascunho: str = "", draft: str = "", etapa: int = 1):
        if draft:
            state = store.wizard_draft(draft, owner_id)
            identifier = draft
        else:
            state = wizard_initial()
            if rascunho and store.draft(owner_id):
                old = store.draft(owner_id)
                state.update({key: old[key] for key in ("document_type", "mapping_id", "fields", "count", "preset", "filter_preset", "generation_mode", "step") if key in old})
            elif job:
                state = reused_state(owner_id, owned_job(owner_id, job))
            elif configuracao:
                old = store.config(configuracao, owner_id)["options"]
                state.update({key: old[key] for key in ("document_type", "mapping_id", "fields", "count", "filter_preset", "generation_mode") if key in old})
            if modelo:
                state["document_type"] = modelo
            identifier = store.create_wizard_draft(owner_id, state)
        if modelo and draft and modelo != state.get("document_type"):
            state.update(document_type=modelo, mapping_id="", fields=[], step=1)
        if state.get("document_type") and not state.get("mapping_id"):
            try:
                state["mapping_id"] = catalog.active_mapping(state["document_type"], owner_id)
            except ServiceError:
                pass
            store.save_wizard_draft(identifier, owner_id, state)
        if preset:
            chosen = find_preset(owner_id, preset)
            if chosen["writing"] and chosen["available"]:
                state["preset"] = preset
                store.save_wizard_draft(identifier, owner_id, state)
                etapa = 4
        etapa = max(1, min(6, etapa))
        if etapa > state.get("step", 1):
            etapa = state.get("step", 1)
        assistant = await run_in_threadpool(assistant_summary) if etapa == 3 else None
        return render(request, "generation.html", page="geracao", **wizard_context(owner_id, state, identifier, etapa, assistant=assistant))

    @router.get("/geracao/mapeamento/previa")
    async def wizard_mapping_preview(request: Request, owner_id: Owner, draft: str):
        state = store.wizard_draft(draft, owner_id)
        current = next((item for item in catalog.document_types([row for row in store.list_models(owner_id) if row["status"] == "published"])
                        if item["id"] == state.get("document_type")), None)
        if not current:
            raise ServiceError(422, "invalid_mapping", "Selecione um documento publicado")
        definition = catalog.definition(current["id"], current["active_mapping"], owner_id)
        fields = [{"id": str(field["field_id"]), "label": field.get("label") or field.get("description") or str(field["field_id"]),
                   "bbox": field.get("bbox"), "elements": field.get("elements") or []} for field in definition["fields"]]
        return render(request, "partials/wizard_pdf.html", current_type=current, definition=definition, fields=fields,
                      selected_fields=[], editable=False)

    @router.post("/geracao/etapa/{step}")
    async def wizard_advance(request: Request, step: int, owner_id: Owner):
        if step not in range(1, 6):
            raise ServiceError(404, "not_found", "Etapa não encontrada")
        form = await request.form()
        identifier = text(form, "draft_id")
        state = store.wizard_draft(identifier, owner_id)
        if step > state.get("step", 1):
            raise ServiceError(422, "validation_error", "Conclua a etapa anterior")
        back = text(form, "direction") == "back"
        if back:
            if step == 2:
                state["fields"] = [str(value) for value in form.getlist("fields")]
            elif step == 3:
                count = parsed(form, "count", int)
                if isinstance(count, int) and count >= 1 and (settings.max_count is None or count <= settings.max_count):
                    state["count"] = count
                if text(form, "generation_mode") in {"traditional", "ai"}:
                    state["generation_mode"] = text(form, "generation_mode")
            elif step == 4 and text(form, "preset"):
                state["preset"] = "original_snapshot" if text(form, "preset") == "original_snapshot" and state.get("writing_snapshot") else preset(owner_id, text(form, "preset"))["id"]
            elif step == 5 and text(form, "filter_preset") in FILTERS:
                state["filter_preset"] = text(form, "filter_preset")
        else:
            if step == 1:
                document_type = text(form, "document_type")
                types = catalog.document_types([row for row in store.list_models(owner_id) if row["status"] == "published"])
                selected = next((item for item in types if item["id"] == document_type), None)
                if not selected:
                    raise ServiceError(422, "validation_error", "Selecione um documento publicado")
                mapping_id = selected["active_mapping"]
                definition = catalog.definition(document_type, mapping_id, owner_id)
                if not definition["fields"]:
                    raise ServiceError(422, "validation_error", "Este mapeamento ainda não possui campos")
                if state["document_type"] != document_type or state["mapping_id"] != mapping_id:
                    state.update(document_type=document_type, mapping_id=mapping_id, fields=[], step=1)
            elif step == 2:
                definition = catalog.definition(state["document_type"], state["mapping_id"], owner_id)
                allowed = {str(field["field_id"]) for field in definition["fields"]}
                selected = [str(value) for value in form.getlist("fields")]
                if not selected or len(selected) != len(set(selected)) or set(selected) - allowed:
                    raise ServiceError(422, "validation_error", "Selecione ao menos um campo existente")
                blocked = sorted(set(selected) & set(unavailable_custom_fields(definition)))
                if blocked:
                    raise ServiceError(422, "validation_error", f"O campo {blocked[0]} não pode ser gerado: {BLOCKED_FIELD.lower()}")
                unsupported = next((field for field in definition["fields"]
                                    if str(field["field_id"]) in selected and field.get("unsupported_reason")), None)
                if unsupported:
                    raise ServiceError(422, "unsupported_field", f"{unsupported.get('label') or unsupported['field_id']}: {unsupported['unsupported_reason']}")
                state["fields"] = selected
            elif step == 3:
                count = parsed(form, "count", int)
                if not isinstance(count, int) or count < 1 or (settings.max_count is not None and count > settings.max_count):
                    message = f"Informe de 1 a {settings.max_count} documentos" if settings.max_count is not None else "Informe ao menos 1 documento"
                    raise ServiceError(422, "validation_error", message)
                mode = text(form, "generation_mode") or "traditional"
                if mode not in {"traditional", "ai"}:
                    raise ServiceError(422, "validation_error", "Selecione um modo de geração válido")
                if mode == "ai":
                    assistant = await run_in_threadpool(assistant_summary)
                    if not assistant["available"]:
                        raise ServiceError(422, "ai_unavailable", f"Geração assistida por IA indisponível: {assistant['detail']}. {assistant['hint']}".strip())
                state.update(count=count, generation_mode=mode)
            elif step == 4:
                if not text(form, "preset"):
                    raise ServiceError(422, "validation_error", "Selecione um estilo de escrita")
                snapshot = text(form, "preset") == "original_snapshot" and state.get("writing_snapshot")
                chosen = None if snapshot else preset(owner_id, text(form, "preset"))
                writing = Writing.model_validate(state["writing_snapshot"] if snapshot else chosen["writing"])
                conflicts = writing_issues(catalog.definition(state["document_type"], state["mapping_id"], owner_id), state["fields"], writing)
                if conflicts:
                    raise ServiceError(422, "writing_incompatible", writing_message(writing, conflicts))
                state["preset"] = "original_snapshot" if snapshot else chosen["id"]
            else:
                value = text(form, "filter_preset")
                if value not in FILTERS:
                    raise ServiceError(422, "validation_error", "Selecione um preset de realismo")
                state["filter_preset"] = value
            state["step"] = max(state.get("step", 1), step + 1)
        store.save_wizard_draft(identifier, owner_id, state)
        return wizard_redirect(request, identifier, max(1, step - 1) if back else step + 1)

    @router.post("/geracao/gerar")
    async def wizard_submit(request: Request, owner_id: Owner):
        identifier = text(await request.form(), "draft_id")
        state = store.claim_wizard_draft(identifier, owner_id)
        if state.get("step", 1) < 6:
            state.pop("submitting", None)
            store.save_wizard_draft(identifier, owner_id, state)
            raise ServiceError(422, "validation_error", "Conclua as cinco etapas antes de gerar")
        try:
            chosen = ({"id": "original_snapshot", "title": state.get("original_preset_name", "Configuração original"),
                       "version": 1, "kind": "original", "writing": state["writing_snapshot"]}
                      if state["preset"] == "original_snapshot" and state.get("writing_snapshot") else preset(owner_id, state["preset"]))
            data = {"name": f"Dataset {state['document_type']}", "document_type": state["document_type"],
                    "mapping_id": state["mapping_id"], "fields": state["fields"], "count": state["count"],
                    "writing": chosen["writing"], "filter_preset": state["filter_preset"],
                    "generation_mode": state.get("generation_mode", "traditional")}
            options = catalog.validate(validated(GenerationOptions, data), owner_id)
            reference = {"id": chosen["id"], "name": chosen["title"], "version": chosen["version"], "kind": chosen["kind"], "adjusted": False}
            row, _ = store.submit(options, request.client.host if request.client else "unknown", owner_id, reference,
                                  catalog.source(options, owner_id))
        except Exception:
            state.pop("submitting", None)
            store.save_wizard_draft(identifier, owner_id, state)
            raise
        state.update(submitting=False, submitted_job=row["id"])
        store.save_wizard_draft(identifier, owner_id, state)
        return Response(headers={"HX-Redirect": "/modelos"})

    @router.get("/geracao/editar-escrita")
    async def wizard_writing_editor(owner_id: PageOwner, draft: str, novo: bool = False):
        store.wizard_draft(draft, owner_id)
        return RedirectResponse(f"/presets/escrita{'/novo' if novo else ''}?retorno=geracao_{draft}", 303)

    @router.post("/geracao/editar-mapeamento")
    async def wizard_mapping_editor(request: Request, owner_id: Owner):
        form = await request.form()
        identifier = text(form, "draft_id")
        state = store.wizard_draft(identifier, owner_id)
        model = model_entry(owner_id, state["document_type"])
        if model.builtin and not model.editable:
            copy = duplicate_model(catalog, store, model.id, owner_id)
            state.update(document_type=copy["id"], mapping_id=copy["mapping_id"], fields=[], step=1)
            store.save_wizard_draft(identifier, owner_id, state)
            model_id = copy["id"]
        else:
            model_id = model.id
        return Response(headers={"HX-Redirect": f"/modelos/{model_id}/editor?draft={identifier}"})

    @router.post("/geracao/rascunho")
    async def generation_draft(request: Request, owner_id: Owner):
        store.save_draft(owner_id, options_state(await request.form()))
        return Response(headers={"HX-Redirect": "/presets/escrita?retorno=geracao"})

    @router.post("/geracao/formulario")
    async def generation_form(request: Request, owner_id: Owner):
        form = await request.form()
        options = options_state(form)
        action = text(form, "acao")
        if action in {"documento", "mapeamento"}:
            options["fields"] = None
        elif action == "modelo":
            options["writing"]["writer"] = None
        elif action == "preset" and text(form, "preset"):
            chosen = preset(owner_id, text(form, "preset"))
            options.update(writing=chosen["writing"], filter_preset=chosen["filter_preset"])
        elif action == "preset":
            options["preset"] = ""
        elif action == "configuracao" and text(form, "configuracao"):
            options = store.config(text(form, "configuracao"), owner_id)["options"]
        return render(request, "partials/generation_sections.html", oob=True,
                      **generation_context(owner_id, options, text(form, "busca")))

    @router.post("/geracao/campos")
    async def generation_fields(request: Request, owner_id: Owner):
        form = await request.form()
        options = options_state(form)
        action = text(form, "acao")
        if action == "todos":
            options["fields"] = None
        elif action == "limpar":
            options["fields"] = []
        return render(request, "partials/field_list.html", oob=True,
                      **generation_context(owner_id, options, text(form, "busca")))

    @router.post("/geracao/resumo")
    async def generation_summary(request: Request, owner_id: Owner):
        return render(request, "partials/summary.html", options=options_state(await request.form()), count_oob=True)

    @router.post("/geracao/configuracoes")
    async def save_configuration(request: Request, owner_id: Owner):
        form = await request.form()
        name = text(form, "nome_configuracao")
        if not name:
            raise ServiceError(422, "validation_error", "Informe um nome para a configuração")
        state = options_state(form)
        state.pop("preset")
        options = catalog.validate(validated(GenerationOptions, state), owner_id)
        store.save_config(owner_id, name[:120], options)
        return render(request, "partials/configurations.html", configs=store.list_configs(owner_id),
                      message="Configuração salva")

    @router.post("/geracao/realismo")
    async def generation_realism(request: Request, owner_id: Owner):
        form = await request.form()
        document_type = text(form, "document_type")
        seed = parsed(form, "seed", int, 42)
        preset_name = text(form, "filter_preset")
        preset_name = preset_name if preset_name in {"light", "medium", "strong"} else "light"
        if not isinstance(seed, int) or not 0 <= seed < 2**32:
            raise ServiceError(422, "validation_error", "Confira o valor de “Seed aleatória”")
        original, filtered, _ = realism_previews(catalog.document(document_type, owner_id).template, preset_name, seed)
        return render(request, "partials/realism_dialog.html", document_type=document_type, seed=seed,
                      preset=preset_name, original=original, filtered=filtered)

    @router.post("/geracao/realismo/aplicar")
    async def apply_realism(request: Request, owner_id: Owner):
        form = await request.form()
        chosen = text(form, "filter_preset")
        return render(request, "partials/realism_select.html", headers={"HX-Trigger-After-Swap": "fechar-modal, realismo-aplicado"},
                      selected=chosen if chosen in FILTERS else "medium", oob=True)

    @router.post("/geracao/jobs")
    async def submit(request: Request, owner_id: Owner):
        state = options_state(await request.form())
        preset_id = state.pop("preset")
        options = catalog.validate(validated(GenerationOptions, state), owner_id)
        chosen = next((item for item in presets(owner_id) if item["id"] == preset_id and item["writing"]), None)
        reference = {"id": chosen["id"], "name": chosen["title"], "version": chosen["version"], "kind": chosen["kind"],
                     "adjusted": chosen["writing"] != options.writing.model_dump()} if chosen else None
        row, _ = store.submit(options, request.client.host if request.client else "unknown", owner_id, reference,
                              catalog.source(options, owner_id))
        return render(request, "partials/job.html", job=store.detail(row))

    @router.get("/geracao/jobs/{job_id}")
    async def job_progress(request: Request, job_id: Identifier, owner_id: Owner):
        return render(request, "partials/job.html", job=store.detail(owned_job(owner_id, job_id)))

    @router.get("/geracao/atividade")
    async def generation_activity(request: Request, owner_id: Owner):
        return render(request, "partials/generation_activity.html", **store.activity(owner_id, time.time() - 120))

    @router.get("/geracao/{draft_id}/amostra.png")
    async def reused_writing_sample(draft_id: Identifier, owner_id: Owner):
        state = store.wizard_draft(draft_id, owner_id)
        if not state.get("writing_snapshot"):
            raise ServiceError(404, "not_found", "Amostra não encontrada")
        writing = Writing.model_validate(state["writing_snapshot"])
        catalog.require_model(writing.model)
        image = await run_in_threadpool(preview_strip, writing, store.root / "previews", 1)
        return Response(image, media_type="image/png", headers={"Cache-Control": "private, max-age=300"})

    @router.get("/geracao/realismo/previa")
    async def wizard_realism_preview(request: Request, owner_id: Owner, draft: str, preset_name: str = "none"):
        state = store.wizard_draft(draft, owner_id)
        if preset_name not in FILTERS:
            raise ServiceError(422, "validation_error", "Preset inválido")
        if preset_name == "none":
            return render(request, "partials/wizard_realism_preview.html", original=None,
                          image=f'/modelos/{state["document_type"]}/previa.png')
        original, filtered, _ = await run_in_threadpool(realism_previews, catalog.document(state["document_type"], owner_id).template, preset_name, 42)
        return render(request, "partials/wizard_realism_preview.html", original=original, image=filtered)

    @router.post("/geracao/jobs/{job_id}/cancelar")
    async def cancel(request: Request, job_id: Identifier, owner_id: Owner):
        owned_job(owner_id, job_id)
        return render(request, "partials/job.html", job=store.detail(store.cancel(job_id)))

    @router.get("/resultados")
    async def results_page(request: Request, owner_id: PageOwner, job: str = "", busca: str = "", status: str = "",
                           modelo: str = "", ordem: str = "recentes", pagina: int = 1):
        if job:
            owned_job(owner_id, job)
            query = urlencode({key: value for key, value in {"busca": busca, "status": status, "modelo": modelo, "ordem": ordem, "pagina": pagina}.items()
                               if value and value not in {"recentes", 1}})
            return RedirectResponse(f"/resultados/{job}" + (f"?{query}" if query else ""), 303)
        return render(request, "results.html", page="resultados", **jobs_context(owner_id, busca, status, document_type=modelo, order=ordem, page=pagina))

    @router.get("/resultados/lista")
    async def results_table(request: Request, owner_id: Owner, busca: str = "", status: str = "", modelo: str = "",
                            ordem: str = "recentes", pagina: int = 1):
        context = jobs_context(owner_id, busca, status, document_type=modelo, order=ordem, page=pagina)
        url = "/resultados" + (f"?{context['detail_query']}" if context["detail_query"] else "")
        return render(request, "partials/jobs_table.html", headers={"HX-Push-Url": url}, **context)

    @router.get("/resultados/{job_id}")
    async def result_detail(request: Request, job_id: Identifier, owner_id: PageOwner, documento: int = 1,
                            busca: str = "", status: str = "", modelo: str = "", ordem: str = "recentes", pagina: int = 1):
        back_query = urlencode({key: value for key, value in {"busca": busca, "status": status, "modelo": modelo, "ordem": ordem, "pagina": pagina}.items()
                                if value and value not in {"recentes", 1}})
        return render(request, "result_detail.html", page="resultados", result=result_context(owner_id, job_id, documento),
                      back_url="/resultados" + (f"?{back_query}" if back_query else ""))

    @router.get("/resultados/{job_id}/documentos/atualizacoes")
    async def result_document_updates(request: Request, job_id: Identifier, owner_id: Owner, documento: int = 1, quantidade: int = 0):
        result = result_context(owner_id, job_id, documento)
        if not result["selected_document"]:
            return Response(headers={"HX-Refresh": "true"})
        return render(request, "partials/result_live.html", result=result, oob_update=True,
                      new_documents=result["documents"][max(0, quantidade):])

    @router.get("/resultados/{job_id}/documentos/{document_number}")
    async def result_document(request: Request, job_id: Identifier, document_number: int, owner_id: Owner):
        result = result_context(owner_id, job_id, document_number)
        if not result["selected_document"]:
            raise ServiceError(404, "not_found", "Documento não encontrado")
        return render(request, "partials/result_workspace.html", result=result)

    @router.post("/resultados/{job_id}/renomear")
    async def result_rename(request: Request, job_id: Identifier, owner_id: Owner):
        name = text(await request.form(), "nome")
        row = store.rename_job(job_id, owner_id, name)
        return render(request, "partials/result_name.html", job=store.detail(row))

    @router.get("/resultados/{job_id}/documentos/{document_number}/paginas/{page_number}.png")
    async def result_pdf_page(job_id: Identifier, document_number: int, page_number: int, owner_id: Owner, largura: int = 1800):
        result = result_context(owner_id, job_id, document_number)
        selected = result["selected_document"]
        if not selected:
            raise ServiceError(404, "not_found", "Documento não encontrado")
        path, _ = store.artifact(owned_job(owner_id, job_id), selected["pdf"]["id"])
        image = await run_in_threadpool(render_pdf_page, path, page_number, max(700, min(largura, 2400)))
        return Response(image, media_type="image/png", headers={"Cache-Control": "private, max-age=300"})

    @router.get("/resultados/{job_id}/documentos/{document_number}/informacoes")
    async def result_document_info(request: Request, job_id: Identifier, document_number: int, owner_id: Owner):
        result = result_context(owner_id, job_id, document_number)
        selected = result["selected_document"]
        if not selected or not selected["json"]:
            raise ServiceError(404, "not_found", "Metadados não encontrados")
        path, artifact = store.artifact(owned_job(owner_id, job_id), selected["json"]["id"])
        if artifact["size"] > JSON_PREVIEW_LIMIT:
            return render(request, "partials/document_info.html", result=result, data=None, raw=None)
        raw = path.read_text(encoding="utf-8", errors="replace")
        try:
            data = json.loads(raw)
        except ValueError:
            data = None
        return render(request, "partials/document_info.html", result=result, data=data, raw=raw)

    @router.get("/resultados/{job_id}/detalhes")
    async def result_metadata(request: Request, job_id: Identifier, owner_id: Owner):
        return render(request, "partials/result_metadata.html", result=result_context(owner_id, job_id))

    @router.get("/resultados/{job_id}/exportar")
    async def result_export_dialog(request: Request, job_id: Identifier, owner_id: Owner):
        result = result_context(owner_id, job_id)
        if not result["archive"]:
            raise ServiceError(409, "not_ready", "O dataset ainda não está disponível")
        return render(request, "partials/result_export.html", result=result)

    @router.get("/resultados/{job_id}/exportar/arquivo")
    async def result_export_file(job_id: Identifier, owner_id: Owner, pdfs: bool = False, jsons: bool = False, nome: str = ""):
        if not pdfs and not jsons:
            raise ServiceError(422, "validation_error", "Selecione PDFs, JSONs ou ambos")
        row = owned_job(owner_id, job_id)
        if row["status"] != "succeeded":
            raise ServiceError(409, "not_ready", "Aguarde a conclusão da geração para exportar o dataset")
        artifacts = store.artifacts(row)
        chosen = [item for item in artifacts if item.get("document_number") and
                  ((pdfs and item["media_type"] == "application/pdf") or (jsons and item["media_type"] == "application/json"))]
        if not chosen:
            manifest = next((item for item in artifacts if item["name"] == "dataset.json"), None)
            if not manifest:
                raise ServiceError(409, "not_ready", "Os documentos ainda não estão disponíveis")
            dataset = json.loads(store.artifact(row, manifest["id"])[0].read_text(encoding="utf-8"))
            names = {entry[key] for entry in dataset.get("documents", []) for key, enabled in (("pdf", pdfs), ("ground_truth", jsons)) if enabled and key in entry}
            chosen = [item for item in artifacts if item["name"] in names]
        if not chosen:
            raise ServiceError(409, "not_ready", "Os documentos ainda não estão disponíveis")
        if pdfs and jsons:
            chosen.extend(item for item in artifacts if item["name"] == "dataset.json")
        descriptor, temporary_name = tempfile.mkstemp(prefix="ocrsd-export-", suffix=".zip", dir=store.root)
        os.close(descriptor)
        archive_path = Path(temporary_name)
        try:
            def build_archive():
                with ZipFile(archive_path, "w", ZIP_DEFLATED) as archive:
                    for item in chosen:
                        path, _ = store.artifact(row, item["id"])
                        archive.write(path, item["name"], ZIP_STORED if item["media_type"] == "application/pdf" else ZIP_DEFLATED)
            await run_in_threadpool(build_archive)
        except Exception:
            archive_path.unlink(missing_ok=True)
            raise
        options = store.detail(row).options
        fallback = f"{options.document_type}-{options.count}-documentos-{datetime.fromtimestamp(row['created_at']).strftime('%Y%m%d')}" if options else "dataset-sintetico"
        cleaned = re.sub(r"[^\w -]", "", nome.strip() or fallback, flags=re.UNICODE).strip()[:100] or fallback
        async def content():
            with archive_path.open("rb") as source:
                while chunk := source.read(1024 * 1024):
                    yield chunk
        return StreamingResponse(content(), media_type="application/zip", background=BackgroundTask(archive_path.unlink, missing_ok=True),
                                 headers={"Content-Disposition": f"attachment; filename*=UTF-8''{quote(cleaned + '.zip')}"})

    @router.get("/resultados/{job_id}/estado")
    async def result_state(request: Request, job_id: Identifier, owner_id: Owner):
        result = result_context(owner_id, job_id)
        if result["job"].status not in ACTIVE:
            return Response(headers={"HX-Refresh": "true"})
        return render(request, "partials/result_workspace.html", result=result)

    @router.post("/resultados/{job_id}/reproduzir")
    async def result_reproduce(request: Request, job_id: Identifier, owner_id: Owner):
        row = owned_job(owner_id, job_id)
        detail = store.detail(row)
        if detail.options is None or detail.options.seed is None:
            raise ServiceError(422, "invalid_configuration", "A configuração original não está disponível para reprodução")
        replay = None
        if detail.options.generation_mode == "ai":
            records = store.records(row)
            if row["status"] != "succeeded" or not records or len(records["documents"]) != detail.options.count:
                raise ServiceError(422, "records_unavailable", "Os registros aceitos pela IA nesta geração não estão disponíveis; "
                                                               "a reprodução exata exige uma geração assistida concluída")
            replay = (row.get("replay_of") or row["id"], records)
        source = store.source(row) or catalog.source(detail.options, owner_id)
        options = catalog.validate(detail.options, owner_id, source["definition"], replay=replay is not None)
        reference = json.loads(row["preset"]) if row.get("preset") else None
        created, _ = store.submit(options, request.client.host if request.client else "unknown", owner_id, reference, source, replay)
        store.rename_job(created["id"], owner_id, f"{detail.name} · reprodução"[:100])
        return Response(headers={"HX-Redirect": f'/resultados/{created["id"]}'})

    @router.post("/resultados/{job_id}/reutilizar")
    async def result_reuse(request: Request, job_id: Identifier, owner_id: Owner):
        state = reused_state(owner_id, owned_job(owner_id, job_id))
        identifier = store.create_wizard_draft(owner_id, state)
        return Response(headers={"HX-Redirect": wizard_url(identifier, 1)})

    @router.get("/resultados/{job_id}/arquivos/{artifact_id}")
    async def result_file(job_id: Identifier, artifact_id: str, owner_id: Owner, download: bool = True, nome: str = ""):
        row = owned_job(owner_id, job_id)
        path, item = store.artifact(row, artifact_id)
        filename = item["name"]
        if item["media_type"] == "application/zip":
            options = store.detail(row).options
            fallback = f"{options.document_type}-{options.count}-documentos-{datetime.fromtimestamp(row['created_at']).strftime('%Y%m%d')}" if options else "dataset-sintetico"
            cleaned = re.sub(r"[^\w -]", "", nome.strip() or fallback, flags=re.UNICODE).strip()[:100]
            filename = (cleaned or fallback) + ".zip"
        return file_response(path, item["media_type"], filename, download)

    @router.get("/resultados/{job_id}/compartilhar")
    async def share_dialog(request: Request, job_id: Identifier, owner_id: Owner):
        job = store.detail(owned_job(owner_id, job_id))
        return render(request, "partials/share_dialog.html", job=job, shares=store.list_shares(job_id, owner_id))

    @router.post("/resultados/{job_id}/compartilhamentos")
    async def create_share(request: Request, job_id: Identifier, owner_id: Owner):
        form = await request.form()
        payload = validated(ShareCreate, {"allow_download": "allow_download" in form,
                                          "expires_in_hours": parsed(form, "expires_in_hours", int, 24)})
        row, token = store.create_share(store.get(job_id), owner_id, payload.allow_download, payload.expires_in_hours)
        link = f'{str(request.base_url).rstrip("/")}/compartilhado/{row["id"]}#token={quote(token, safe="")}'
        return render(request, "partials/share_created.html", job=store.detail(owned_job(owner_id, job_id)), link=link,
                      share=row, shares=store.list_shares(job_id, owner_id), oob=True)

    @router.post("/compartilhamentos/{share_id}/revogar")
    async def revoke_share(request: Request, share_id: Identifier, owner_id: Owner):
        row = store.revoke_share(share_id, owner_id)
        return render(request, "partials/shares.html", job=store.detail(owned_job(owner_id, row["job_id"])),
                      shares=store.list_shares(row["job_id"], owner_id))

    def preset_cards(owner_id: str, busca: str, aba: str, modelo: str, ordem: str) -> tuple[list[dict], dict]:
        items = presets(owner_id)
        groups = {"todos": items, "meus": [item for item in items if item["kind"] == "meu"],
                  "compartilhados": [item for item in items if item["kind"] == "compartilhado" or item["shared"]],
                  "padroes": [item for item in items if item["kind"] == "padrao"]}
        selected = groups.get(aba, items)
        if modelo:
            selected = [item for item in selected if item["writing"] and item["writing"]["model"] == modelo]
        needle = busca.casefold().strip()
        if needle:
            selected = [item for item in selected if needle in " ".join((item["title"], item["name"], item["description"], item["summary"])).casefold()]
        if ordem == "nome":
            selected = sorted(selected, key=lambda item: item["title"].casefold())
        else:
            key = "created_at" if ordem == "recentes" else "updated_at"
            selected = sorted(selected, key=lambda item: (item[key] is None, -(item[key] or 0)))
        return selected, {key: len(value) for key, value in groups.items()}

    def gallery_context(owner_id: str, busca: str, aba: str, modelo: str, ordem: str, pagina: int, retorno: str) -> dict:
        aba = aba if aba in PRESET_TABS else "todos"
        ordem = ordem if ordem in PRESET_SORTS else "modificados"
        modelo = modelo if modelo in MODEL_INFO else ""
        retorno = valid_return(owner_id, retorno)
        cards, counts = preset_cards(owner_id, busca[:120], aba, modelo, ordem)
        pagina = max(1, pagina)
        visible = cards[(pagina - 1) * PRESET_PAGE_SIZE:pagina * PRESET_PAGE_SIZE]
        draft = store.wizard_draft(retorno.removeprefix("geracao_"), owner_id) if retorno.startswith("geracao_") else store.draft(owner_id) if retorno else None
        return {"cards": [{**item, "thumbnail": preset_thumbnail(item)} for item in visible], "counts": counts, "busca": busca[:120],
                "aba": aba, "modelo": modelo, "ordem": ordem, "pagina": pagina, "retorno": retorno,
                "more": len(cards) > pagina * PRESET_PAGE_SIZE, "total": len(cards), "selected": (draft or {}).get("preset", ""),
                "tabs": PRESET_TABS, "sorts": PRESET_SORTS, "model_info": MODEL_INFO}

    def use_link(identifier: str, retorno: str) -> str:
        if retorno.startswith("geracao_"):
            return f"/geracao?draft={retorno.removeprefix('geracao_')}&preset={identifier}"
        return f"/geracao?rascunho=1&preset={identifier}" if retorno == "geracao" else f"/geracao?preset={identifier}"

    def valid_return(owner_id: str, value: str) -> str:
        if value == "geracao":
            return value
        if re.fullmatch(r"geracao_[0-9a-f]{32}", value):
            store.wizard_draft(value.removeprefix("geracao_"), owner_id)
            return value
        return ""

    templates.env.globals["use_link"] = use_link

    def model_cards() -> list[dict]:
        return [{**MODEL_INFO[item["id"]], "id": item["id"], "available": item["available"], "reason": item["reason"]}
                for item in catalog.models()]

    def editor_writing(form: FormData) -> dict:
        writing = writing_state(form)
        if text(form, "writer_mode") == "aleatorio":
            writing["writer"] = None
        return writing

    def preset_editor(request: Request, owner_id: str, entry: dict | None, writing: dict, retorno: str, rename: bool = False) -> HTMLResponse:
        mode = "novo" if entry is None else "editar" if entry["editable"] else "visualizar"
        grants = store.preset_grants(entry["id"]) if entry and entry["editable"] else 0
        return render(request, "preset_editor.html", page="presets", entry=entry, writing=writing, mode=mode, models=model_cards(),
                      info=MODEL_INFO, samples=PREVIEW_SAMPLES, retorno=valid_return(owner_id, retorno), rename=rename,
                      grants=grants, link=None, characters=NEURAL_CHARACTERS)

    @router.get("/presets")
    async def presets_redirect():
        return RedirectResponse("/presets/escrita", 303)

    @router.get("/presets/escrita")
    async def preset_gallery(request: Request, owner_id: PageOwner, busca: str = "", aba: str = "todos", modelo: str = "",
                             ordem: str = "modificados", pagina: int = 1, retorno: str = ""):
        context = gallery_context(owner_id, busca, aba, modelo, ordem, pagina, retorno)
        target = request.headers.get("hx-target") if not request.headers.get("hx-history-restore-request") else None
        if target == "grade-presets":
            return render(request, "partials/preset_grid.html", **context)
        if target and target.startswith("mais-presets"):
            return render(request, "partials/preset_cards.html", **context)
        return render(request, "preset_gallery.html", page="presets", **context)

    @router.get("/presets/escrita/novo")
    async def preset_new(request: Request, owner_id: PageOwner, modelo: str = "font", retorno: str = ""):
        model = modelo if modelo in MODEL_INFO and model_status(modelo)[0] else "font"
        return preset_editor(request, owner_id, None, model_defaults(model), retorno)

    @router.get("/presets/escrita/modelos")
    async def preset_models(request: Request, owner_id: PageOwner):
        return render(request, "writing_models.html", page="presets", models=model_cards())

    @router.post("/presets/escrita/controles")
    async def preset_controls(request: Request, owner_id: Owner):
        form = await request.form()
        writing = editor_writing(form)
        model = writing["model"] if writing["model"] in MODEL_INFO else "font"
        writing = model_defaults(model, writing) if text(form, "acao") == "padroes" else adapt_to_model(writing, model, text(form, "modelo_atual"))
        return render(request, "partials/writing_controls.html", writing=writing, info=MODEL_INFO, characters=NEURAL_CHARACTERS)

    @router.post("/presets/escrita/previa")
    async def preset_preview(request: Request, owner_id: Owner):
        form = await request.form()
        writing = validated(Writing, editor_writing(form))
        available, reason = model_status(writing.model)
        variations = parsed(form, "variacoes", int, 1)
        variations = variations if variations in {1, 2, 3} else 1
        if not available:
            return render(request, "partials/playground.html", rows=[], unavailable=reason, variations=variations, writing=writing)
        texts = [sample for _, sample in PREVIEW_SAMPLES]
        columns = [await run_in_threadpool(render_writing_samples, writing, texts, variation) for variation in range(variations)]
        rows = [{"label": label, "text": sample, "cells": [
                    {**column[index], "image": "data:image/png;base64," + base64.b64encode(column[index]["image"]).decode() if column[index]["image"] else None}
                    for column in columns]}
                for index, (label, sample) in enumerate(PREVIEW_SAMPLES)]
        return render(request, "partials/playground.html", rows=rows, unavailable=None, variations=variations, writing=writing)

    @router.get("/presets/escrita/convite/{preset_id}")
    async def preset_invitation(request: Request, preset_id: Identifier, owner_id: PageOwner):
        return render(request, "preset_invitation.html", page="presets", preset_id=preset_id)

    @router.post("/presets/escrita/convite/{preset_id}")
    async def preset_accept(request: Request, preset_id: Identifier, owner_id: Owner):
        store.accept_preset(preset_id, text(await request.form(), "token"), owner_id)
        return Response(status_code=204, headers={"HX-Redirect": f"/presets/escrita/{preset_id}"})

    @router.post("/presets/escrita/salvar")
    async def preset_save(request: Request, owner_id: Owner):
        form = await request.form()
        name, description, source = text(form, "preset_name"), text(form, "description"), text(form, "id")
        if not 1 <= len(name) <= 120:
            raise ServiceError(422, "validation_error", "Informe um nome com até 120 caracteres")
        if len(description) > 500:
            raise ServiceError(422, "validation_error", "A descrição aceita até 500 caracteres")
        writing = validated(Writing, editor_writing(form))
        catalog.require_model(writing.model)
        selected_return = valid_return(owner_id, text(form, "retorno"))
        retorno = f"?retorno={selected_return}" if selected_return else ""
        if text(form, "acao") == "salvar" and source:
            current = store.preset(source, owner_id)
            row = store.save_preset(owner_id, name, writing.model_dump(), current["filter_preset"], bool(current["shareable"]), source, description)
            response = toast(request, f"Preset salvo · versão {row['version']}")
            response.headers["HX-Trigger"] = json.dumps({"preset-salvo": {"version": row["version"]}})
            return response
        original = find_preset(owner_id, source) if source else None
        if original and name == original["title"]:
            name = f"Cópia de {name}"[:120]
        row = store.save_preset(owner_id, name, writing.model_dump(), original["filter_preset"] if original else "none", False, None, description)
        return Response(headers={"HX-Redirect": f'/presets/escrita/{row["id"]}{retorno}'})

    @router.get("/presets/escrita/{preset_id}")
    async def preset_page(request: Request, preset_id: str, owner_id: PageOwner, retorno: str = "", renomear: str = ""):
        entry = find_preset(owner_id, preset_id)
        return preset_editor(request, owner_id, entry, entry["writing"] or model_defaults("font"), retorno, bool(renomear))

    @router.get("/presets/escrita/{preset_id}/amostra.png")
    async def preset_sample(preset_id: str, owner_id: Owner, linhas: int = 4):
        entry = preset(owner_id, preset_id)
        rows = 1 if linhas == 1 else 4
        image = await run_in_threadpool(preview_strip, Writing.model_validate(entry["writing"]), store.root / "previews", rows)
        return Response(image, media_type="image/png", headers={"Cache-Control": "private, max-age=86400"})

    @router.post("/presets/escrita/{preset_id}/duplicar")
    async def preset_duplicate(request: Request, preset_id: str, owner_id: Owner):
        entry = find_preset(owner_id, preset_id)
        if not entry["writing"]:
            raise ServiceError(422, "preset_mixed", "Presets mistos do desktop não podem ser duplicados na web")
        row = store.save_preset(owner_id, f'Cópia de {entry["title"]}'[:120], entry["writing"], entry["filter_preset"], False, None, entry["description"])
        selected_return = valid_return(owner_id, text(await request.form(), "retorno"))
        retorno = f"&retorno={selected_return}" if selected_return else ""
        return Response(headers={"HX-Redirect": f'/presets/escrita/{row["id"]}?renomear=1{retorno}'})

    @router.post("/presets/escrita/{preset_id}/excluir")
    async def preset_delete(request: Request, preset_id: str, owner_id: Owner):
        entry = find_preset(owner_id, preset_id)
        if entry["kind"] == "padrao":
            raise ServiceError(409, "preset_standard", "Presets padrão não podem ser excluídos")
        if entry["kind"] == "meu":
            store.delete_preset(preset_id, owner_id)
            message = "Preset excluído"
        else:
            store.leave_preset(preset_id, owner_id)
            message = "Preset removido da sua coleção"
        if text(await request.form(), "origem") == "editor":
            return Response(headers={"HX-Redirect": "/presets/escrita"})
        return toast(request, message)

    @router.post("/presets/escrita/{preset_id}/compartilhamento")
    async def preset_share(request: Request, preset_id: Identifier, owner_id: Owner):
        action = text(await request.form(), "acao")
        link = None
        if action == "privado":
            store.unshare_preset(preset_id, owner_id)
        elif action in {"link", "novo-link"}:
            token = store.share_preset(preset_id, owner_id)
            link = f'{str(request.base_url).rstrip("/")}/presets/escrita/convite/{preset_id}#token={quote(token, safe="")}'
        else:
            raise ServiceError(422, "validation_error", "Ação de compartilhamento inválida")
        entry = find_preset(owner_id, preset_id)
        return render(request, "partials/preset_share.html", entry=entry, link=link, grants=store.preset_grants(preset_id),
                      message="Link criado; copie-o agora" if link else "Preset voltou a ser privado")

    @router.get("/ferramentas/realismo")
    async def realism_page(request: Request, owner_id: PageOwner):
        return render(request, "realism_page.html", page="ferramentas", types=catalog.document_types(store.list_models(owner_id)))

    @router.post("/ferramentas/realismo")
    async def realism_compare(request: Request, owner_id: Owner):
        form = await request.form()
        preset_name = text(form, "filter_preset")
        seed = parsed(form, "seed", int, 42)
        if preset_name not in {"light", "medium", "strong"}:
            raise ServiceError(422, "validation_error", "Escolha uma intensidade de realismo")
        if not isinstance(seed, int) or not 0 <= seed < 2**32:
            raise ServiceError(422, "validation_error", "Confira o valor de “Seed aleatória”")
        original, filtered, _ = realism_previews(catalog.document(text(form, "document_type"), owner_id).template, preset_name, seed)
        return render(request, "partials/realism.html", preset=preset_name, original=original, filtered=filtered)

    @router.get("/compartilhado/{share_id}")
    async def shared_page(request: Request, share_id: Identifier):
        token = request.cookies.get(SHARE_COOKIE)
        if not token:
            return render(request, "shared.html", share_id=share_id, pending=True)
        share = store.share(share_id, token)
        row = store.get(share["job_id"])
        artifacts = store.artifacts(row)
        return render(request, "shared.html", share_id=share_id, pending=False, share=share, job=store.detail(row),
                      pdf=next((item for item in artifacts if item["media_type"] == "application/pdf"), None),
                      archive=next((item for item in artifacts if item["media_type"] == "application/zip"), None))

    @router.post("/compartilhado/{share_id}/acesso")
    async def shared_access(request: Request, share_id: Identifier):
        token = text(await request.form(), "token")
        share = store.share(share_id, token)
        response = Response(status_code=204, headers={"HX-Refresh": "true"})
        set_cookie(response, request, SHARE_COOKIE, token, max(1, int(share["expires_at"] - time.time())),
                   f"/compartilhado/{share_id}")
        return response

    @router.get("/compartilhado/{share_id}/arquivos/{artifact_id}")
    async def shared_file(request: Request, share_id: Identifier, artifact_id: str, download: bool = False):
        share = store.share(share_id, request.cookies.get(SHARE_COOKIE) or "")
        if download and not share["allow_download"]:
            raise ServiceError(401, "download_forbidden", "Este link permite somente visualização")
        path, item = store.artifact(store.get(share["job_id"]), artifact_id)
        return file_response(path, item["media_type"], item["name"], download)

    app.include_router(router)
    return interface_error
