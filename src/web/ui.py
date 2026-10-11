import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Callable
from urllib.parse import quote
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, ValidationError
from starlette.datastructures import FormData, UploadFile

from src.web.models import GenerationOptions, Identifier, ModelUpdate, ShareCreate, Writing
from src.web.services import (Catalog, ServiceError, create_model, duplicate_model, handwriting_previews,
                              realism_previews, render_preview, save_mapping)
from src.web.settings import Settings

ROOT = Path(__file__).resolve().parent
WORKSPACE_COOKIE = "ocrsd_workspace"
SHARE_COOKIE = "ocrsd_share"
FILTERS = {"none": "Sem realismo", "light": "Leve", "medium": "Médio", "strong": "Forte"}
STATUS = {"queued": "Na fila", "running": "Em execução", "succeeded": "Concluído", "failed": "Falhou",
          "cancelled": "Cancelado", "expired": "Expirado"}
ACTIVE = {"queued", "running"}
DATA_TYPES = ("text", "person_name", "integer", "float", "date", "datetime", "time", "enum", "checkbox",
              "digit_sequence", "text_sequence")
SAMPLES = (("Nome", "Ana Silva Santos"), ("Data", "15092026"), ("UF", "SP"))
MAPPING_ID = re.compile(r"^[A-Za-z0-9_-]{1,100}$")
JSON_PREVIEW_LIMIT = 2 * 1024 * 1024
LABELS = {"name": "Nome", "count": "Quantidade de documentos", "seed": "Seed aleatória", "fields": "Campos",
          "document_type": "Modelo", "mapping_id": "Mapeamento", "filter_preset": "Realismo visual",
          "model": "Modelo de escrita", "intensity": "Variação da escrita", "writer": "Identidade do escritor",
          "bias": "Regularidade", "color": "Cor", "width": "Espessura", "pressure": "Pressão", "dropout": "Falhas",
          "text_policy": "Caracteres não suportados", "label": "Nome do modelo", "description": "Descrição",
          "expires_in_hours": "Validade do link"}
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
    return {"name": text(form, "name"), "document_type": text(form, "document_type"),
            "mapping_id": text(form, "mapping_id"), "fields": [str(item) for item in form.getlist("fields")],
            "count": parsed(form, "count", int, 1), "seed": parsed(form, "seed", int, 0),
            "writing": writing_state(form), "filter_preset": text(form, "filter_preset") or "none"}


def set_cookie(response: Response, request: Request, name: str, value: str, max_age: int, path: str = "/") -> None:
    response.set_cookie(name, value, max_age=max_age, path=path, httponly=True, samesite="lax",
                        secure=request.url.scheme == "https")


def mount_ui(app: FastAPI, settings: Settings, store, catalog: Catalog) -> Callable[[Request, int, str], Response]:
    templates = Jinja2Templates(directory=ROOT / "templates")
    templates.env.filters.update(data=format_date, tamanho=format_size)
    templates.env.globals.update(FILTERS=FILTERS, STATUS=STATUS, ACTIVE=ACTIVE, DATA_TYPES=DATA_TYPES,
                                 VERSION=app.version)
    app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")
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

    def presets(owner_id: str) -> list[dict]:
        items = [{"id": item.id, "label": item.label, "writing": item.writing.model_dump() if item.writing else None,
                  "filter_preset": item.filter_preset, "available": item.available, "reason": item.reason, "builtin": True}
                 for item in catalog.presets()]
        items += [{"id": row["id"], "label": row["name"], "writing": json.loads(row["writing"]),
                   "filter_preset": row["filter_preset"], "available": True, "reason": None, "builtin": False}
                  for row in store.list_presets(owner_id)]
        return items

    def preset(owner_id: str, identifier: str) -> dict:
        found = next((item for item in presets(owner_id) if item["id"] == identifier), None)
        if not found or not found["writing"]:
            raise ServiceError(404, "not_found", "Preset não encontrado")
        if not found["available"]:
            raise ServiceError(422, "model_unavailable", found["reason"] or "Preset indisponível")
        return found

    def model_entry(owner_id: str, model_id: str):
        model = next((item for item in catalog.document_models(store.list_models(owner_id)) if item.id == model_id), None)
        if model is None:
            raise ServiceError(404, "not_found", "Modelo não encontrado")
        return model

    def default_options() -> dict:
        return {"name": "Novo dataset sintético", "document_type": "", "mapping_id": "", "fields": None,
                "count": min(10, settings.max_count), "seed": 42, "writing": Writing().model_dump(), "filter_preset": "medium"}

    def generation_context(owner_id: str, options: dict, search: str = "") -> dict:
        types = catalog.document_types(store.list_models(owner_id))
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
        return {"options": options, "types": types, "current_type": current, "fields": fields, "search": search,
                "models": catalog.models(), "presets": presets(owner_id), "configs": store.list_configs(owner_id),
                "max_count": settings.max_count}

    def owned_job(owner_id: str, job_id: str) -> dict:
        return store.authorize_owner(store.get(job_id), owner_id)

    def viewer_context(row: dict, artifact: dict | None) -> dict:
        content = None
        if artifact and artifact["media_type"] == "application/json":
            path, _ = store.artifact(row, artifact["id"])
            content = path.read_text(encoding="utf-8", errors="replace") if artifact["size"] <= JSON_PREVIEW_LIMIT else None
        return {"artifact": artifact, "content": content}

    def result_context(owner_id: str, job_id: str) -> dict:
        row = owned_job(owner_id, job_id)
        artifacts, shares, problem = [], [], None
        if row["status"] == "succeeded":
            try:
                artifacts = store.artifacts(row)
                shares = store.list_shares(job_id, owner_id)
            except ServiceError as exc:
                problem = exc.message
        files = [item for item in artifacts if item["media_type"] != "application/zip"]
        first = next((item for item in files if item["media_type"] == "application/pdf"), files[0] if files else None)
        return {"job": store.detail(row), "files": files, "shares": shares, "problem": problem,
                "archive": next((item for item in artifacts if item["media_type"] == "application/zip"), None),
                "viewer": viewer_context(row, first)}

    def jobs_context(owner_id: str, search: str, status: str, selected: str = "") -> dict:
        status = status if status in STATUS else ""
        jobs = store.list_jobs(owner_id, search[:120], status or None)
        return {"jobs": jobs, "search": search[:120], "status": status, "selected": selected,
                "polling": any(job["status"] in ACTIVE for job in jobs)}

    @router.get("/")
    async def index():
        return RedirectResponse("/modelos", 303)

    @router.post("/sessao")
    async def adopt_session(request: Request):
        form = await request.form()
        token = text(form, "token")
        store.workspace(token)
        response = Response(status_code=204, headers={"HX-Refresh": "true"})
        set_cookie(response, request, WORKSPACE_COOKIE, token, 400 * 86400)
        return response

    @router.get("/modelos")
    async def models_page(request: Request, owner_id: PageOwner):
        return render(request, "models.html", page="modelos", models=catalog.document_models(store.list_models(owner_id)),
                      max_upload=settings.max_upload_bytes)

    @router.get("/modelos/novo")
    async def new_model(request: Request, owner_id: Owner):
        return render(request, "partials/model_form.html", max_upload=settings.max_upload_bytes)

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

    @router.post("/modelos/{model_id}/copia")
    async def duplicate(model_id: Identifier, owner_id: Owner):
        row = duplicate_model(catalog, store, model_id, owner_id)
        return Response(headers={"HX-Redirect": f'/modelos/{row["id"]}/editor'})

    @router.get("/modelos/{model_id}/previa.png")
    async def model_preview(model_id: Identifier, owner_id: Owner):
        return Response(render_preview(catalog.document(model_id, owner_id).template), media_type="image/png",
                        headers={"Cache-Control": "private, max-age=300"})

    @router.get("/modelos/{model_id}/editor")
    async def editor_page(request: Request, model_id: Identifier, owner_id: PageOwner):
        model = model_entry(owner_id, model_id)
        if model.builtin and not model.editable:
            return RedirectResponse("/modelos", 303)
        definition = catalog.definition(model.id, model.mapping_id, None if model.builtin else owner_id)
        return render(request, "editor.html", page="modelos", model=model, data_types=DATA_TYPES,
                      editor={"definition": definition, "mapping_id": model.mapping_id, "preview": f"/modelos/{model.id}/previa.png",
                              "source": f"/modelos/{model.id}/mapeamento.json"})

    @router.get("/modelos/{model_id}/mapeamento.json")
    async def mapping_json(model_id: Identifier, owner_id: Owner):
        model = model_entry(owner_id, model_id)
        return JSONResponse(catalog.definition(model.id, model.mapping_id, None if model.builtin else owner_id))

    @router.post("/modelos/{model_id}/mapeamento")
    async def save(request: Request, model_id: Identifier, owner_id: Owner):
        form = await request.form()
        model = model_entry(owner_id, model_id)
        try:
            definition = json.loads(text(form, "definition"))
        except ValueError as exc:
            raise ServiceError(422, "invalid_mapping", "O mapeamento enviado não é um JSON válido") from exc
        if not model.builtin:
            metadata = validated(ModelUpdate, {"label": text(form, "label"), "description": text(form, "description"),
                                               "status": text(form, "status") or "draft"})
        save_mapping(catalog, store, model_id, owner_id, definition)
        if not model.builtin:
            store.update_model(model_id, owner_id, metadata.label, metadata.description, metadata.status)
        return toast(request, "Mapeamento salvo")

    @router.post("/modelos/{model_id}/excluir")
    async def delete(model_id: Identifier, owner_id: Owner):
        store.delete_model(model_id, owner_id)
        return Response(headers={"HX-Redirect": "/modelos"})

    @router.get("/geracao")
    async def generation_page(request: Request, owner_id: PageOwner, modelo: str = "", job: str = "", configuracao: str = ""):
        options = default_options()
        if job:
            source = store.detail(owned_job(owner_id, job)).options
            if source:
                options = {**source.model_dump(), "name": f"{source.name} — nova execução"}
        elif configuracao:
            options = store.config(configuracao, owner_id)["options"]
        elif modelo:
            options["document_type"] = modelo
        return render(request, "generation.html", page="geracao", **generation_context(owner_id, options))

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

    @router.get("/geracao/presets")
    async def generation_presets(request: Request, owner_id: Owner):
        return render(request, "partials/preset_options.html", presets=presets(owner_id))

    @router.post("/geracao/configuracoes")
    async def save_configuration(request: Request, owner_id: Owner):
        form = await request.form()
        name = text(form, "nome_configuracao")
        if not name:
            raise ServiceError(422, "validation_error", "Informe um nome para a configuração")
        options = catalog.validate(validated(GenerationOptions, options_state(form)), owner_id)
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
        options = catalog.validate(validated(GenerationOptions, options_state(await request.form())), owner_id)
        row, _ = store.submit(options, request.client.host if request.client else "unknown", owner_id)
        return render(request, "partials/job.html", job=store.detail(row))

    @router.get("/geracao/jobs/{job_id}")
    async def job_progress(request: Request, job_id: Identifier, owner_id: Owner):
        return render(request, "partials/job.html", job=store.detail(owned_job(owner_id, job_id)))

    @router.post("/geracao/jobs/{job_id}/cancelar")
    async def cancel(request: Request, job_id: Identifier, owner_id: Owner):
        owned_job(owner_id, job_id)
        return render(request, "partials/job.html", job=store.detail(store.cancel(job_id)))

    @router.get("/resultados")
    async def results_page(request: Request, owner_id: PageOwner, job: str = "", busca: str = "", status: str = ""):
        try:
            selected = result_context(owner_id, job) if job else None
        except ServiceError:
            selected, job = None, ""
        return render(request, "results.html", page="resultados", result=selected, **jobs_context(owner_id, busca, status, job))

    @router.get("/resultados/lista")
    async def results_table(request: Request, owner_id: Owner, busca: str = "", status: str = "", job: str = ""):
        return render(request, "partials/jobs_table.html", **jobs_context(owner_id, busca, status, job))

    @router.get("/resultados/{job_id}")
    async def result_detail(request: Request, job_id: Identifier, owner_id: Owner, busca: str = "", status: str = ""):
        return render(request, "partials/result.html", result=result_context(owner_id, job_id), oob=True,
                      **jobs_context(owner_id, busca, status, job_id))

    @router.get("/resultados/{job_id}/visualizar/{artifact_id}")
    async def result_viewer(request: Request, job_id: Identifier, artifact_id: str, owner_id: Owner):
        row = owned_job(owner_id, job_id)
        _, artifact = store.artifact(row, artifact_id)
        files = [item for item in store.artifacts(row) if item["media_type"] != "application/zip"]
        return render(request, "partials/viewer.html", job=store.detail(row), files=files, oob=True,
                      viewer=viewer_context(row, artifact))

    @router.get("/resultados/{job_id}/arquivos/{artifact_id}")
    async def result_file(job_id: Identifier, artifact_id: str, owner_id: Owner, download: bool = True):
        path, item = store.artifact(owned_job(owner_id, job_id), artifact_id)
        return file_response(path, item["media_type"], item["name"], download)

    @router.get("/resultados/{job_id}/compartilhar")
    async def share_dialog(request: Request, job_id: Identifier, owner_id: Owner):
        return render(request, "partials/share_dialog.html", job=store.detail(owned_job(owner_id, job_id)))

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

    @router.get("/presets")
    async def presets_page(request: Request, owner_id: PageOwner):
        return render(request, "presets.html", page="presets", writing=Writing().model_dump(), models=catalog.models(),
                      presets=presets(owner_id), types=catalog.document_types(store.list_models(owner_id)))

    @router.get("/presets/laboratorio")
    async def presets_dialog(request: Request, owner_id: Owner):
        return render(request, "partials/lab_dialog.html", writing=Writing().model_dump(), models=catalog.models(),
                      presets=presets(owner_id))

    @router.post("/presets/formulario")
    async def preset_controls(request: Request, owner_id: Owner):
        form = await request.form()
        writing = writing_state(form)
        if text(form, "acao") == "modelo":
            writing["writer"] = None
        if text(form, "preset"):
            writing = preset(owner_id, text(form, "preset"))["writing"]
        return render(request, "partials/lab_controls.html", writing=writing, models=catalog.models())

    @router.post("/presets/amostras")
    async def preset_samples(request: Request, owner_id: Owner):
        writing = validated(Writing, writing_state(await request.form()))
        catalog.require_model(writing.model)
        images = handwriting_previews(writing, [sample for _, sample in SAMPLES])
        return render(request, "partials/samples.html", samples=[(label, image) for (label, _), image in zip(SAMPLES, images)])

    @router.post("/presets")
    async def save_preset(request: Request, owner_id: Owner):
        form = await request.form()
        name = text(form, "preset_name")
        if not name:
            raise ServiceError(422, "validation_error", "Informe um nome para o preset")
        writing = validated(Writing, writing_state(form))
        catalog.require_model(writing.model)
        store.save_preset(owner_id, name[:120], writing.model_dump(), "none", False)
        return render(request, "partials/preset_list.html", headers={"HX-Trigger": "presets-atualizados"},
                      presets=presets(owner_id), message="Preset salvo")

    @router.post("/presets/{preset_id}/excluir")
    async def delete_preset(request: Request, preset_id: Identifier, owner_id: Owner):
        store.delete_preset(preset_id, owner_id)
        return render(request, "partials/preset_list.html", headers={"HX-Trigger": "presets-atualizados"},
                      presets=presets(owner_id), message="Preset excluído")

    @router.post("/presets/realismo")
    async def preset_realism(request: Request, owner_id: Owner):
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
