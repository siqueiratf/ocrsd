import logging
from typing import Annotated

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from starlette.exceptions import HTTPException

from src.web.jobs import JobStore
from src.web.models import Capabilities, DocumentTypeResponse, ErrorResponse, GenerationOptions, Health, Identifier, JobResponse, MappingResponse, PresetResponse, Results, SubmittedJob
from src.web.services import Catalog, ServiceError
from src.web.settings import Settings


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    catalog = Catalog(settings)
    store = JobStore(settings)
    errors = {code: {"model": ErrorResponse} for code in (401, 404, 405, 409, 410, 411, 413, 415, 422, 429, 500, 503)}
    app = FastAPI(title="OCRSD — documentos sintéticos", version="1.1.0", docs_url="/api/docs", redoc_url=None,
                  openapi_url="/api/openapi.json", responses=errors)
    app.state.store = store
    security = HTTPBearer(auto_error=False, description="Chave retornada ao criar o job. Não compartilhar.")

    def error(status: int, code: str, message: str) -> JSONResponse:
        return JSONResponse(status_code=status, content={"error": {"code": code, "message": message}})

    @app.exception_handler(ServiceError)
    async def service_error(request: Request, exc: ServiceError):
        return error(exc.status, exc.code, exc.message)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        return error(422, "validation_error", "Parâmetros inválidos; confira os campos e limites no contrato da API")

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException):
        return error(exc.status_code, "http_error", "Recurso ou operação indisponível")

    @app.exception_handler(Exception)
    async def unexpected_error(request: Request, exc: Exception):
        logging.getLogger("ocrsd.api").exception("Falha na API", exc_info=exc)
        return error(500, "internal_error", "Falha interna; tente novamente mais tarde")

    @app.middleware("http")
    async def guard(request: Request, call_next):
        if request.method == "POST":
            content_type = request.headers.get("content-type", "").split(";")[0]
            if content_type != "application/json":
                return error(415, "content_type", "Use application/json")
            length = request.headers.get("content-length")
            if length is None:
                return error(411, "length_required", "Informe Content-Length")
            if not length.isdigit() or int(length) > 32768:
                return error(413, "too_large", "Corpo da requisição excede o limite")
            body = bytearray()
            async for chunk in request.stream():
                body.extend(chunk)
                if len(body) > 32768:
                    return error(413, "too_large", "Corpo da requisição excede o limite")
            request._body = bytes(body)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Cache-Control"] = "no-store"
        return response

    def authorized(job_id: Identifier, credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(security)]) -> dict:
        if credentials is None:
            raise ServiceError(401, "unauthorized", "Informe a chave de acesso do job")
        return store.authorize(job_id, credentials.credentials)

    @app.get("/api/v1/health", response_model=Health, tags=["Operação"])
    def health():
        ready = store.ready()
        return Health(status="ok" if ready else "degraded", worker_ready=ready)

    @app.get("/api/v1/document-types", response_model=list[DocumentTypeResponse], tags=["Catálogo"])
    def document_types():
        return [{"id": item.key, "label": item.label,
                 "mappings": [{"id": path.stem, "label": path.stem.replace("_", " ")} for path in item.mappings]}
                for item in catalog.types.values()]

    @app.get("/api/v1/document-types/{document_type}/template", tags=["Catálogo"], response_class=FileResponse,
             responses={200: {"content": {"application/pdf": {"schema": {"type": "string", "format": "binary"}}}}})
    def template(document_type: Identifier):
        return FileResponse(catalog.document(document_type).template, media_type="application/pdf", filename="template.pdf")

    @app.get("/api/v1/document-types/{document_type}/mappings/{mapping_id}", response_model=MappingResponse, tags=["Catálogo"])
    def mapping(document_type: Identifier, mapping_id: Identifier):
        definition = catalog.definition(document_type, mapping_id)
        fields = [{"field_id": str(field["field_id"]), "label": field.get("label") or field.get("description") or str(field["field_id"]),
                   "data_type": field.get("data_type", "text"), "bbox": field["bbox"]} for field in definition["fields"]]
        return MappingResponse(id=mapping_id, definition=definition, fields=fields)

    @app.get("/api/v1/capabilities", response_model=Capabilities, tags=["Catálogo"])
    def capabilities():
        return Capabilities(models=catalog.models(), filters=["none", "light", "medium", "strong"],
                            max_count=settings.max_count, retention_hours=settings.retention_hours)

    @app.get("/api/v1/writing-presets", response_model=list[PresetResponse], tags=["Catálogo"])
    def writing_presets():
        return catalog.presets()

    @app.post("/api/v1/generation/validate", response_model=GenerationOptions, tags=["Geração"])
    def validate(options: GenerationOptions):
        return catalog.validate(options)

    @app.post("/api/v1/jobs", response_model=SubmittedJob, status_code=202, tags=["Geração"])
    def submit(options: GenerationOptions, request: Request):
        catalog.validate(options)
        row, token = store.submit(options, request.client.host if request.client else "unknown")
        return SubmittedJob(**store.response(row).model_dump(), token=token)

    @app.get("/api/v1/jobs/{job_id}", response_model=JobResponse, tags=["Jobs"])
    def job(row: Annotated[dict, Depends(authorized)]):
        return store.response(row)

    @app.post("/api/v1/jobs/{job_id}/cancel", response_model=JobResponse, tags=["Jobs"])
    def cancel(row: Annotated[dict, Depends(authorized)]):
        return store.response(store.cancel(row["id"]))

    @app.get("/api/v1/jobs/{job_id}/results", response_model=Results, tags=["Resultados"])
    def results(row: Annotated[dict, Depends(authorized)]):
        return Results(artifacts=store.artifacts(row))

    @app.get("/api/v1/jobs/{job_id}/artifacts/{artifact_id}", response_class=FileResponse, tags=["Resultados"],
             responses={200: {"content": {mime: {"schema": {"type": "string", "format": "binary"}} for mime in ("application/pdf", "application/json", "application/zip")}}})
    def artifact(artifact_id: str,
                 row: Annotated[dict, Depends(authorized)], download: bool = True):
        path, item = store.artifact(row, artifact_id)
        return FileResponse(path, media_type=item["media_type"], filename=item["name"], content_disposition_type="attachment" if download else "inline")

    return app
