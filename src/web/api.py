import base64
import logging
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Header, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from starlette.exceptions import HTTPException

from src.web.jobs import JobStore
from src.web.models import Artifact, Capabilities, DocumentModelResponse, DocumentTypeResponse, ErrorResponse, GenerationConfigCreate, GenerationConfigResponse, GenerationOptions, HandwritingPreviewRequest, HandwritingPreviewResponse, Health, Identifier, JobDetail, JobResponse, MappingResponse, MappingUpdate, ModelCreate, ModelUpdate, PresetResponse, RealismPreviewRequest, RealismPreviewResponse, Results, ShareCreate, ShareCreated, ShareResponse, SharedResult, SubmittedJob, WorkspaceResponse, WritingPresetCreate, WritingPresetStored
from src.web.services import Catalog, ServiceError, create_model as create_document_model, custom_model, duplicate_model as duplicate_document_model, handwriting_previews, realism_previews, render_preview, save_mapping
from src.web.settings import Settings
from src.web.ui import file_response, mount_ui

CONTENT_SECURITY_POLICY = (b"default-src 'self'; img-src 'self' data:; frame-src 'self'; object-src 'none'; "
                           b"base-uri 'self'; form-action 'self'; frame-ancestors 'self'")
FORM_TYPES = {"application/x-www-form-urlencoded", "multipart/form-data"}


class GuardMiddleware:
    def __init__(self, app: Any, settings: Settings):
        self.app = app
        self.settings = settings

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = {key.decode("latin1").lower(): value.decode("latin1") for key, value in scope.get("headers", [])}
        guarded_receive = receive
        path = scope.get("path", "")
        api = path.startswith("/api/")
        if not api and scope["method"] not in {"GET", "HEAD"} and headers.get("hx-request") != "true":
            await JSONResponse(status_code=403, content={"error": {"code": "forbidden", "message": "Operação permitida somente pela interface"}})(scope, receive, send)
            return
        if scope["method"] in {"POST", "PUT", "PATCH"}:
            content_type = headers.get("content-type", "").split(";")[0]
            empty = not api and not content_type and headers.get("content-length") == "0"
            if not empty and content_type not in ({"application/json"} if api else FORM_TYPES):
                await JSONResponse(status_code=415, content={"error": {"code": "content_type", "message": "Use application/json" if api else "Tipo de conteúdo não suportado"}})(scope, receive, send)
                return
            length = headers.get("content-length")
            if length is None:
                await JSONResponse(status_code=411, content={"error": {"code": "length_required", "message": "Informe Content-Length"}})(scope, receive, send)
                return
            if path in {"/api/v1/models", "/modelos"}:
                maximum = self.settings.max_upload_bytes * 2
            elif path.endswith("/mapping") or path.endswith("/mapeamento"):
                maximum = 131072 if api else 393216
            else:
                maximum = 32768
            if not length.isdigit() or int(length) > maximum:
                await JSONResponse(status_code=413, content={"error": {"code": "too_large", "message": "Corpo da requisição excede o limite"}})(scope, receive, send)
                return
            body = bytearray()
            more_body = True
            while more_body:
                message = await receive()
                if message["type"] == "http.disconnect":
                    return
                body.extend(message.get("body", b""))
                if len(body) > maximum:
                    await JSONResponse(status_code=413, content={"error": {"code": "too_large", "message": "Corpo da requisição excede o limite"}})(scope, receive, send)
                    return
                more_body = message.get("more_body", False)
            delivered = False

            async def replay() -> dict:
                nonlocal delivered
                if delivered:
                    return {"type": "http.disconnect"}
                delivered = True
                return {"type": "http.request", "body": bytes(body), "more_body": False}

            guarded_receive = replay

        async def secure_send(message: dict) -> None:
            if message["type"] == "http.response.start":
                response_headers = list(message.get("headers", []))
                names = {name.lower() for name, _ in response_headers}
                response_headers.extend([
                    (b"x-content-type-options", b"nosniff"),
                    (b"referrer-policy", b"no-referrer"),
                    (b"x-frame-options", b"SAMEORIGIN"),
                ])
                if b"cache-control" not in names:
                    response_headers.append((b"cache-control", b"no-store"))
                content_type = dict(response_headers).get(b"content-type", b"")
                if not api and content_type.startswith(b"text/html"):
                    response_headers.append((b"content-security-policy", CONTENT_SECURITY_POLICY))
                message["headers"] = response_headers
            await send(message)

        await self.app(scope, guarded_receive, secure_send)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    store = JobStore(settings)
    catalog = Catalog(settings)
    errors = {code: {"model": ErrorResponse} for code in (401, 404, 405, 409, 410, 411, 413, 415, 422, 429, 500, 503)}
    app = FastAPI(title="OCRSD — documentos sintéticos", version="1.2.0", docs_url="/api/docs", redoc_url=None,
                  openapi_url="/api/openapi.json", responses=errors)
    app.add_middleware(GuardMiddleware, settings=settings)
    app.state.store = store
    interface_error = mount_ui(app, settings, store, catalog)
    security = HTTPBearer(auto_error=False, description="Chave privada do job ou do compartilhamento.")

    def error(request: Request, status: int, code: str, message: str) -> Response:
        if not request.url.path.startswith("/api/"):
            return interface_error(request, status, message)
        return JSONResponse(status_code=status, content={"error": {"code": code, "message": message}})

    @app.exception_handler(ServiceError)
    async def service_error(request: Request, exc: ServiceError):
        return error(request, exc.status, exc.code, exc.message)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        return error(request, 422, "validation_error", "Parâmetros inválidos; confira os campos e limites no contrato da API")

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException):
        return error(request, exc.status_code, "http_error", "Recurso ou operação indisponível")

    @app.exception_handler(Exception)
    async def unexpected_error(request: Request, exc: Exception):
        logging.getLogger("ocrsd.api").exception("Falha na API", exc_info=exc)
        return error(request, 500, "internal_error", "Falha interna; tente novamente mais tarde")

    async def workspace_optional(x_workspace_token: Annotated[str | None, Header()] = None) -> str | None:
        return store.workspace(x_workspace_token) if x_workspace_token else None

    async def workspace_required(x_workspace_token: Annotated[str | None, Header()] = None) -> str:
        if not x_workspace_token:
            raise ServiceError(401, "workspace_required", "Informe a chave do espaço de trabalho")
        return store.workspace(x_workspace_token)

    async def authorized(job_id: Identifier,
                   credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(security)],
                   x_workspace_token: Annotated[str | None, Header()] = None) -> dict:
        if x_workspace_token:
            return store.authorize_owner(store.get(job_id), store.workspace(x_workspace_token))
        if credentials is None:
            raise ServiceError(401, "unauthorized", "Informe a chave de acesso do job")
        return store.authorize(job_id, credentials.credentials)

    def mapping_response(document_type: str, mapping_id: str, owner_id: str | None = None) -> MappingResponse:
        definition = catalog.definition(document_type, mapping_id, owner_id)
        fields = [{"field_id": str(field["field_id"]), "label": field.get("label") or field.get("description") or str(field["field_id"]),
                   "data_type": field.get("data_type", "text"), "bbox": field["bbox"]} for field in definition["fields"]]
        return MappingResponse(id=mapping_id, definition=definition, fields=fields)

    def preset_response(row: dict) -> WritingPresetStored:
        from src.web.models import Writing

        return WritingPresetStored(id=row["id"], label=row["name"], writing=Writing.model_validate_json(row["writing"]),
                                   filter_preset=row["filter_preset"], available=True, reason=None, builtin=False,
                                   shareable=bool(row["shareable"]), created_at=row["created_at"], updated_at=row["updated_at"])

    def share_response(row: dict) -> ShareResponse:
        return ShareResponse(id=row["id"], job_id=row["job_id"], allow_download=bool(row["allow_download"]),
                             created_at=row["created_at"], expires_at=row["expires_at"], revoked_at=row["revoked_at"])

    @app.get("/api/v1/health", response_model=Health, tags=["Operação"])
    async def health():
        ready = store.ready()
        return Health(status="ok" if ready else "degraded", worker_ready=ready)

    @app.post("/api/v1/workspaces", response_model=WorkspaceResponse, status_code=201, tags=["Espaço de trabalho"])
    async def create_workspace():
        identifier, token = store.create_workspace()
        return WorkspaceResponse(id=identifier, token=token)

    @app.get("/api/v1/document-types", response_model=list[DocumentTypeResponse], tags=["Catálogo"])
    async def document_types(owner_id: Annotated[str | None, Depends(workspace_optional)]):
        return catalog.document_types(store.list_models(owner_id) if owner_id else [])

    @app.get("/api/v1/models", response_model=list[DocumentModelResponse], tags=["Modelos"])
    async def models(owner_id: Annotated[str, Depends(workspace_required)]):
        return catalog.document_models(store.list_models(owner_id))

    @app.post("/api/v1/models", response_model=DocumentModelResponse, status_code=201, tags=["Modelos"])
    async def create_model(payload: ModelCreate, owner_id: Annotated[str, Depends(workspace_required)]):
        if not payload.filename.lower().endswith(".pdf"):
            raise ServiceError(422, "invalid_pdf", "Envie um arquivo PDF")
        content = payload.pdf_base64.partition(",")[2] if payload.pdf_base64.startswith("data:") else payload.pdf_base64
        try:
            pdf = base64.b64decode(content, validate=True)
        except ValueError:
            raise ServiceError(422, "invalid_pdf", "Conteúdo PDF inválido")
        return custom_model(create_document_model(store, settings, owner_id, payload.label, payload.description,
                                                  payload.mapping_id, payload.filename, pdf, payload.mapping))

    @app.post("/api/v1/models/{model_id}/duplicate", response_model=DocumentModelResponse, status_code=201, tags=["Modelos"])
    async def duplicate_model(model_id: Identifier, owner_id: Annotated[str, Depends(workspace_required)]):
        return custom_model(duplicate_document_model(catalog, store, model_id, owner_id))

    @app.put("/api/v1/models/{model_id}", response_model=DocumentModelResponse, tags=["Modelos"])
    async def update_model(model_id: Identifier, payload: ModelUpdate, owner_id: Annotated[str, Depends(workspace_required)]):
        return custom_model(store.update_model(model_id, owner_id, payload.label, payload.description, payload.status))

    @app.delete("/api/v1/models/{model_id}", tags=["Modelos"])
    async def delete_model(model_id: Identifier, owner_id: Annotated[str, Depends(workspace_required)]):
        store.delete_model(model_id, owner_id)
        return {"deleted": True}

    @app.get("/api/v1/document-types/{document_type}/template", tags=["Catálogo"], response_class=Response,
             responses={200: {"content": {"application/pdf": {"schema": {"type": "string", "format": "binary"}}}}})
    async def template(document_type: Identifier, owner_id: Annotated[str | None, Depends(workspace_optional)]):
        return file_response(catalog.document(document_type, owner_id).template, "application/pdf", "template.pdf")

    @app.get("/api/v1/models/{model_id}/preview", tags=["Modelos"], response_class=Response,
             responses={200: {"content": {"image/png": {"schema": {"type": "string", "format": "binary"}}}}})
    async def model_preview(model_id: Identifier, owner_id: Annotated[str | None, Depends(workspace_optional)]):
        return Response(render_preview(catalog.document(model_id, owner_id).template), media_type="image/png")

    @app.get("/api/v1/document-types/{document_type}/mappings/{mapping_id}", response_model=MappingResponse, tags=["Catálogo"])
    async def mapping(document_type: Identifier, mapping_id: Identifier, owner_id: Annotated[str | None, Depends(workspace_optional)]):
        return mapping_response(document_type, mapping_id, owner_id)

    @app.put("/api/v1/models/{model_id}/mapping", response_model=MappingResponse, tags=["Modelos"])
    async def update_mapping(model_id: Identifier, payload: MappingUpdate, owner_id: Annotated[str, Depends(workspace_required)]):
        mapping_id = save_mapping(catalog, store, model_id, owner_id, payload.definition)
        return mapping_response(model_id, mapping_id, None if model_id in catalog.types else owner_id)

    @app.get("/api/v1/capabilities", response_model=Capabilities, tags=["Catálogo"])
    async def capabilities():
        return Capabilities(models=catalog.models(), filters=["none", "light", "medium", "strong"],
                            max_count=settings.max_count, retention_hours=settings.retention_hours)

    @app.get("/api/v1/writing-presets", response_model=list[PresetResponse], tags=["Catálogo"])
    async def writing_presets():
        return catalog.presets()

    @app.get("/api/v1/presets", response_model=list[WritingPresetStored], tags=["Presets"])
    async def presets(owner_id: Annotated[str, Depends(workspace_required)]):
        return [WritingPresetStored(**item.model_dump(), builtin=True) for item in catalog.presets()] + [preset_response(row) for row in store.list_presets(owner_id)]

    @app.post("/api/v1/presets", response_model=WritingPresetStored, status_code=201, tags=["Presets"])
    async def create_preset(payload: WritingPresetCreate, owner_id: Annotated[str, Depends(workspace_required)]):
        catalog.require_model(payload.writing.model)
        return preset_response(store.save_preset(owner_id, payload.name, payload.writing.model_dump(), payload.filter_preset, payload.shareable))

    @app.put("/api/v1/presets/{preset_id}", response_model=WritingPresetStored, tags=["Presets"])
    async def update_preset(preset_id: Identifier, payload: WritingPresetCreate, owner_id: Annotated[str, Depends(workspace_required)]):
        return preset_response(store.save_preset(owner_id, payload.name, payload.writing.model_dump(), payload.filter_preset, payload.shareable, preset_id))

    @app.delete("/api/v1/presets/{preset_id}", tags=["Presets"])
    async def delete_preset(preset_id: Identifier, owner_id: Annotated[str, Depends(workspace_required)]):
        store.delete_preset(preset_id, owner_id)
        return {"deleted": True}

    @app.post("/api/v1/handwriting/preview", response_model=HandwritingPreviewResponse, tags=["Presets"])
    async def handwriting_preview(payload: HandwritingPreviewRequest, owner_id: Annotated[str, Depends(workspace_required)]):
        catalog.require_model(payload.writing.model)
        return HandwritingPreviewResponse(images=handwriting_previews(payload.writing, payload.texts))

    @app.post("/api/v1/realism/preview", response_model=RealismPreviewResponse, tags=["Presets"])
    async def realism_preview(payload: RealismPreviewRequest, owner_id: Annotated[str, Depends(workspace_required)]):
        original, filtered, metadata = realism_previews(catalog.document(payload.document_type, owner_id).template, payload.preset, payload.seed)
        return RealismPreviewResponse(original=original, filtered=filtered, metadata=metadata)

    @app.get("/api/v1/configurations", response_model=list[GenerationConfigResponse], tags=["Configurações"])
    async def configurations(owner_id: Annotated[str, Depends(workspace_required)]):
        return store.list_configs(owner_id)

    @app.post("/api/v1/configurations", response_model=GenerationConfigResponse, status_code=201, tags=["Configurações"])
    async def create_configuration(payload: GenerationConfigCreate, owner_id: Annotated[str, Depends(workspace_required)]):
        catalog.validate(payload.options, owner_id)
        return store.save_config(owner_id, payload.name, payload.options)

    @app.delete("/api/v1/configurations/{configuration_id}", tags=["Configurações"])
    async def delete_configuration(configuration_id: Identifier, owner_id: Annotated[str, Depends(workspace_required)]):
        store.delete_config(configuration_id, owner_id)
        return {"deleted": True}

    @app.post("/api/v1/generation/validate", response_model=GenerationOptions, tags=["Geração"])
    async def validate(options: GenerationOptions, owner_id: Annotated[str | None, Depends(workspace_optional)]):
        return catalog.validate(options, owner_id)

    @app.post("/api/v1/jobs", response_model=SubmittedJob, status_code=202, tags=["Geração"])
    async def submit(options: GenerationOptions, request: Request, owner_id: Annotated[str | None, Depends(workspace_optional)]):
        catalog.validate(options, owner_id)
        row, token = store.submit(options, request.client.host if request.client else "unknown", owner_id)
        return SubmittedJob(**store.response(row).model_dump(), token=token)

    @app.get("/api/v1/jobs", response_model=list[JobDetail], tags=["Resultados"])
    async def jobs(owner_id: Annotated[str, Depends(workspace_required)], query: Annotated[str, Query(max_length=120)] = "", status: str | None = None):
        if status and status not in {"queued", "running", "succeeded", "failed", "cancelled", "expired"}:
            raise ServiceError(422, "invalid_status", "Status inválido")
        return store.list_jobs(owner_id, query, status)

    @app.get("/api/v1/jobs/{job_id}", response_model=JobDetail, tags=["Jobs"])
    async def job(row: Annotated[dict, Depends(authorized)]):
        return store.detail(row)

    @app.post("/api/v1/jobs/{job_id}/cancel", response_model=JobResponse, tags=["Jobs"])
    async def cancel(row: Annotated[dict, Depends(authorized)]):
        return store.response(store.cancel(row["id"]))

    @app.get("/api/v1/jobs/{job_id}/results", response_model=Results, tags=["Resultados"])
    async def results(row: Annotated[dict, Depends(authorized)]):
        return Results(artifacts=store.artifacts(row))

    @app.get("/api/v1/jobs/{job_id}/artifacts/{artifact_id}", response_class=Response, tags=["Resultados"],
             responses={200: {"content": {mime: {"schema": {"type": "string", "format": "binary"}} for mime in ("application/pdf", "application/json", "application/zip")}}})
    async def artifact(artifact_id: str, row: Annotated[dict, Depends(authorized)], download: bool = True):
        path, item = store.artifact(row, artifact_id)
        return file_response(path, item["media_type"], item["name"], download)

    @app.get("/api/v1/jobs/{job_id}/shares", response_model=list[ShareResponse], tags=["Compartilhamento"])
    async def shares(job_id: Identifier, owner_id: Annotated[str, Depends(workspace_required)]):
        return [share_response(row) for row in store.list_shares(job_id, owner_id)]

    @app.post("/api/v1/jobs/{job_id}/shares", response_model=ShareCreated, status_code=201, tags=["Compartilhamento"])
    async def create_share(job_id: Identifier, payload: ShareCreate, owner_id: Annotated[str, Depends(workspace_required)]):
        row, token = store.create_share(store.get(job_id), owner_id, payload.allow_download, payload.expires_in_hours)
        return ShareCreated(**share_response(row).model_dump(), token=token)

    @app.delete("/api/v1/shares/{share_id}", response_model=ShareResponse, tags=["Compartilhamento"])
    async def revoke_share(share_id: Identifier, owner_id: Annotated[str, Depends(workspace_required)]):
        return share_response(store.revoke_share(share_id, owner_id))

    @app.get("/api/v1/shared/{share_id}", response_model=SharedResult, tags=["Compartilhamento"])
    async def shared(share_id: Identifier, credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(security)]):
        if credentials is None:
            raise ServiceError(401, "unauthorized", "Informe a chave do compartilhamento")
        share = store.share(share_id, credentials.credentials)
        row = store.get(share["job_id"])
        return SharedResult(share=share_response(share), job=store.detail(row), artifacts=[Artifact(**item) for item in store.artifacts(row)])

    @app.get("/api/v1/shared/{share_id}/artifacts/{artifact_id}", response_class=Response, tags=["Compartilhamento"])
    async def shared_artifact(share_id: Identifier, artifact_id: str,
                        credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(security)], download: bool = False):
        if credentials is None:
            raise ServiceError(401, "unauthorized", "Informe a chave do compartilhamento")
        share = store.share(share_id, credentials.credentials)
        if download and not share["allow_download"]:
            raise ServiceError(401, "download_forbidden", "Este link permite somente visualização")
        path, item = store.artifact(store.get(share["job_id"]), artifact_id)
        return file_response(path, item["media_type"], item["name"], download)

    return app
