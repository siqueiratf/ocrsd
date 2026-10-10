from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Identifier = Annotated[str, Field(pattern=r"^[A-Za-z0-9_-]{1,100}$")]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, strict=True)


class Writing(StrictModel):
    model: Literal["font", "sjvasquez", "aaron1259", "wpvnteam"] = "font"
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
    document_type: Identifier
    mapping_id: Identifier
    fields: list[Annotated[str, Field(min_length=1, max_length=100)]] = Field(min_length=1, max_length=200)
    count: int = Field(default=1, ge=1, le=100, strict=True)
    seed: int = Field(default=42, ge=0, le=2**32 - 1, strict=True)
    writing: Writing = Field(default_factory=Writing)
    filter_preset: Literal["none", "light", "medium", "strong"] = "none"


class Resource(BaseModel):
    id: str
    label: str


class DocumentTypeResponse(Resource):
    mappings: list[Resource]


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
    max_count: int
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
