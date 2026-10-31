from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.fonts import FONT_FAMILY_KEYS

Identifier = Annotated[str, Field(pattern=r"^[A-Za-z0-9_-]{1,100}$")]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, strict=True)


class Writing(StrictModel):
    model: Literal["font", "sjvasquez", "aaron1259", "wpvnteam"] = "font"
    font_family: Literal[*FONT_FAMILY_KEYS] = "caveat"
    handwriting: bool = True
    text_handwriting: bool = True
    intensity: float = Field(default=0.65, ge=0, le=1)
    writer: int | None = Field(default=None, ge=0, le=511, strict=True)
    bias: float = Field(default=0.75, ge=0, le=3)
    color: str = Field(default="#193787", pattern=r"^#[0-9a-fA-F]{6}$")
    width: float = Field(default=1.5, gt=0, le=40)
    pressure: float = Field(default=0.25, ge=0, le=1)
    dropout: float = Field(default=0, ge=0, le=1)
    text_policy: Literal["normalize", "strict"] = "normalize"

    @model_validator(mode="after")
    def validate_writer(self) -> Writing:
        if self.model == "sjvasquez" and self.writer is not None and self.writer > 12:
            raise ValueError("sjvasquez aceita estilos de 0 a 12")
        return self


class GenerationOptions(StrictModel):
    name: Annotated[str, Field(min_length=1, max_length=120)] = "Dataset sintético"
    document_type: Identifier
    mapping_id: Identifier | None = None
    fields: list[Annotated[str, Field(min_length=1, max_length=100)]] = Field(min_length=1, max_length=200)
    count: int = Field(default=1, ge=1, strict=True)
    seed: int | None = Field(default=None, ge=0, le=2**32 - 1, strict=True)
    writing: Writing = Field(default_factory=Writing)
    filter_preset: Literal["none", "light", "medium", "strong"] = "none"
    generation_mode: Literal["traditional", "ai"] = "traditional"


class Resource(BaseModel):
    id: str
    label: str


class Orgao(BaseModel):
    key: str
    sigla: str
    nome: str


class DocumentTypeResponse(Resource):
    mappings: list[Resource]
    orgao: Orgao | None = None
    documentacao: str | None = None
    categoria: str | None = None


class FieldResponse(BaseModel):
    field_id: str
    label: str
    data_type: str
    bbox: list[int]


class MappingResponse(BaseModel):
    id: str
    definition: dict[str, Any]
    fields: list[FieldResponse]


class ModelResponse(Resource):
    available: bool
    reason: str | None = None


class PresetResponse(Resource):
    writing: Writing | None
    filter_preset: Literal["none", "light", "medium", "strong"] = "none"
    available: bool
    reason: str | None = None


class Capabilities(BaseModel):
    models: list[ModelResponse]
    filters: list[str]
    max_count: int | None
    retention_hours: int


class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorDetail


class JobResponse(BaseModel):
    id: str
    status: Literal["queued", "running", "succeeded", "failed", "cancelled", "expired"]
    completed: int
    total: int
    created_at: float
    updated_at: float
    expires_at: float
    cancel_requested: bool
    error: str | None


class JobDetail(JobResponse):
    name: str
    options: GenerationOptions | None


class AnalysisCreate(StrictModel):
    page: Annotated[int, Field(ge=1, le=500)] = 1


class AnalysisApply(StrictModel):
    accepted: Annotated[list[Annotated[str, Field(min_length=1, max_length=100)]], Field(max_length=500)]


class AnalysisResponse(BaseModel):
    id: str
    model_id: str
    page: int
    status: Literal["queued", "running", "succeeded", "failed"]
    progress: int
    stage: str
    error: str | None
    provenance: dict[str, Any] | None
    proposal: dict[str, Any] | None
    created_at: float
    updated_at: float
    started_at: float | None
    finished_at: float | None


class AssistantStatus(BaseModel):
    available: bool
    provider: str
    model: str | None
    detail: str


class WorkspaceResponse(BaseModel):
    id: str
    token: str


class DocumentModelResponse(BaseModel):
    id: str
    label: str
    description: str
    status: Literal["draft", "published"]
    version: int
    field_count: int
    mapping_id: str
    builtin: bool
    editable: bool
    orgao: Orgao | None = None
    documentacao: str | None = None
    categoria: str | None = None
    created_at: float | None = None
    updated_at: float | None = None


class ModelCreate(StrictModel):
    label: Annotated[str, Field(min_length=1, max_length=120)]
    description: Annotated[str, Field(max_length=500)] = ""
    filename: Annotated[str, Field(min_length=1, max_length=180)]
    pdf_base64: Annotated[str, Field(min_length=8)]
    mapping_id: Identifier = "default"
    mapping: dict[str, Any] | None = None


class ModelUpdate(StrictModel):
    label: Annotated[str, Field(min_length=1, max_length=120)]
    description: Annotated[str, Field(max_length=500)] = ""
    status: Literal["draft", "published"] = "draft"


class MappingUpdate(StrictModel):
    definition: dict[str, Any]


class ModelPromote(StrictModel):
    key: Annotated[str, Field(min_length=1, max_length=100)]
    label: Annotated[str, Field(min_length=1, max_length=120)]
    description: Annotated[str, Field(max_length=500)] = ""
    orgao: Annotated[str, Field(min_length=1, max_length=50)]
    documentacao: Annotated[str, Field(max_length=500)] = ""
    categoria: Annotated[str, Field(max_length=120)] = ""


class SemanticDefinitionCreate(StrictModel):
    key: Annotated[str, Field(pattern=r"^custom(?:\.[a-z][a-z0-9_]*){1,5}$", max_length=100)]
    label: Annotated[str, Field(min_length=1, max_length=100)]
    description: Annotated[str, Field(min_length=1, max_length=500)]
    data_type: Literal["text", "integer", "boolean", "date", "time"]
    generator: dict[str, Any]


class WritingPresetCreate(StrictModel):
    name: Annotated[str, Field(min_length=1, max_length=120)]
    writing: Writing
    filter_preset: Literal["none", "light", "medium", "strong"] = "none"
    shareable: bool = False


class WritingPresetStored(PresetResponse):
    builtin: bool = False
    shareable: bool = False
    created_at: float | None = None
    updated_at: float | None = None


class GenerationConfigCreate(StrictModel):
    name: Annotated[str, Field(min_length=1, max_length=120)]
    options: GenerationOptions


class GenerationConfigResponse(GenerationConfigCreate):
    id: str
    created_at: float
    updated_at: float


class ShareCreate(StrictModel):
    allow_download: bool = True
    expires_in_hours: int = Field(default=24, ge=1, le=720, strict=True)


class ShareResponse(BaseModel):
    id: str
    job_id: str
    allow_download: bool
    created_at: float
    expires_at: float
    revoked_at: float | None


class ShareCreated(ShareResponse):
    token: str


class SharedResult(BaseModel):
    share: ShareResponse
    job: JobDetail
    artifacts: list[Artifact]


class HandwritingPreviewRequest(StrictModel):
    writing: Writing
    texts: list[Annotated[str, Field(min_length=1, max_length=80)]] = Field(default=["Ana Silva Santos", "15092026", "SP"], min_length=1, max_length=6)


class HandwritingPreviewResponse(BaseModel):
    images: list[str]


class RealismPreviewRequest(StrictModel):
    document_type: Identifier
    preset: Literal["light", "medium", "strong"] = "medium"
    seed: int = Field(default=42, ge=0, le=2**32 - 1, strict=True)


class RealismPreviewResponse(BaseModel):
    original: str
    filtered: str
    metadata: dict[str, Any]


class SubmittedJob(JobResponse):
    token: str


class Artifact(BaseModel):
    id: str
    name: str
    media_type: str
    size: int


class Results(BaseModel):
    artifacts: list[Artifact]


class Health(BaseModel):
    status: Literal["ok", "degraded"]
    worker_ready: bool
