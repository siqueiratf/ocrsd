from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from src.web.api import create_app
from src.web.models import DocumentModelResponse, GenerationConfigResponse, GenerationOptions, JobDetail, JobResponse, ShareCreated, ShareResponse, SubmittedJob, WorkspaceResponse, Writing, WritingPresetStored
from src.web.services import Catalog
from src.web.settings import Settings

ROOT = Path(__file__).resolve().parents[1]


def event(lines: list[str]) -> list[dict]:
    return [{"listen": "test", "script": {"type": "text/javascript", "exec": lines}}]


def build_collection() -> dict:
    with TemporaryDirectory() as temporary:
        defaults = {name: field.default for name, field in Settings.model_fields.items()}
        settings = Settings(**{**defaults, "storage": Path(temporary), "neural_enabled": False})
        app = create_app(settings)
        schema = app.openapi()
        catalog = Catalog(settings)
        health = {"status": "degraded", "worker_ready": False}
        types = [{"id": item.key, "label": item.label,
                  "mappings": [{"id": path.stem, "label": path.stem.replace("_", " ")} for path in item.mappings]}
                 for item in catalog.types.values()]
        capabilities = {"models": catalog.models(), "filters": ["none", "light", "medium", "strong"],
                        "max_count": settings.max_count, "retention_hours": settings.retention_hours}
        built_in_presets = [preset.model_dump() for preset in catalog.presets()]
        document = types[0]
        mapping_id = document["mappings"][0]["id"]
        definition = catalog.definition(document["id"], mapping_id)
        mapping = {"id": mapping_id, "definition": definition,
                   "fields": [{"field_id": str(field["field_id"]),
                               "label": field.get("label") or field.get("description") or str(field["field_id"]),
                               "data_type": field.get("data_type", "text"), "bbox": field["bbox"]}
                              for field in definition["fields"]]}
    options = GenerationOptions(name="DNV — fluxo Postman", document_type=document["id"], mapping_id=mapping_id,
                                fields=[field["field_id"] for field in mapping["fields"][:3]])
    timestamp = 1791648000.0
    job = JobResponse(id="0" * 32, status="queued", completed=0, total=1, created_at=timestamp,
                      updated_at=timestamp, expires_at=timestamp + 86400, cancel_requested=False, error=None)
    detail = JobDetail(**job.model_dump(), name=options.name, options=options)
    model = DocumentModelResponse(id=document["id"], label=document["label"], description="Modelo incluído no catálogo OCRSD",
                                  status="published", version=1, field_count=len(mapping["fields"]), mapping_id=mapping_id,
                                  builtin=True, editable=False)
    preset = WritingPresetStored(id="preset_exemplo", label="Escrita azul natural", writing=Writing(), filter_preset="none",
                                 available=True, reason=None, builtin=False, shareable=False, created_at=timestamp, updated_at=timestamp)
    configuration = GenerationConfigResponse(id="1" * 32, name="Configuração DNV", options=options, created_at=timestamp, updated_at=timestamp)
    share = ShareResponse(id="2" * 32, job_id=job.id, allow_download=True, created_at=timestamp,
                          expires_at=timestamp + 3600, revoked_at=None)
    examples: dict[tuple[str, str], tuple[int, object, str]] = {
        ("/api/v1/health", "get"): (200, health, "application/json"),
        ("/api/v1/workspaces", "post"): (201, WorkspaceResponse(id="3" * 32, token="CHAVE_PRIVADA_DO_ESPACO").model_dump(), "application/json"),
        ("/api/v1/document-types", "get"): (200, types, "application/json"),
        ("/api/v1/models", "get"): (200, [model.model_dump()], "application/json"),
        ("/api/v1/models", "post"): (201, {**model.model_dump(), "id": "4" * 32, "builtin": False, "status": "draft"}, "application/json"),
        ("/api/v1/models/{model_id}/duplicate", "post"): (201, {**model.model_dump(), "id": "4" * 32, "builtin": False, "status": "draft"}, "application/json"),
        ("/api/v1/models/{model_id}", "put"): (200, {**model.model_dump(), "id": "4" * 32, "builtin": False, "status": "draft"}, "application/json"),
        ("/api/v1/models/{model_id}", "delete"): (200, {"deleted": True}, "application/json"),
        ("/api/v1/document-types/{document_type}/mappings/{mapping_id}", "get"): (200, mapping, "application/json"),
        ("/api/v1/models/{model_id}/mapping", "put"): (200, mapping, "application/json"),
        ("/api/v1/capabilities", "get"): (200, capabilities, "application/json"),
        ("/api/v1/writing-presets", "get"): (200, built_in_presets, "application/json"),
        ("/api/v1/presets", "get"): (200, [preset.model_dump()], "application/json"),
        ("/api/v1/presets", "post"): (201, preset.model_dump(), "application/json"),
        ("/api/v1/presets/{preset_id}", "put"): (200, preset.model_dump(), "application/json"),
        ("/api/v1/presets/{preset_id}", "delete"): (200, {"deleted": True}, "application/json"),
        ("/api/v1/handwriting/preview", "post"): (200, {"images": ["data:image/png;base64,EXEMPLO"]}, "application/json"),
        ("/api/v1/realism/preview", "post"): (200, {"original": "data:image/jpeg;base64,EXEMPLO", "filtered": "data:image/jpeg;base64,EXEMPLO", "metadata": {"preset": "medium"}}, "application/json"),
        ("/api/v1/configurations", "get"): (200, [configuration.model_dump()], "application/json"),
        ("/api/v1/configurations", "post"): (201, configuration.model_dump(), "application/json"),
        ("/api/v1/configurations/{configuration_id}", "delete"): (200, {"deleted": True}, "application/json"),
        ("/api/v1/generation/validate", "post"): (200, options.model_dump(), "application/json"),
        ("/api/v1/jobs", "post"): (202, SubmittedJob(**job.model_dump(), token="CHAVE_PRIVADA_DO_JOB").model_dump(), "application/json"),
        ("/api/v1/jobs", "get"): (200, [detail.model_dump()], "application/json"),
        ("/api/v1/jobs/{job_id}", "get"): (200, detail.model_dump(), "application/json"),
        ("/api/v1/jobs/{job_id}/cancel", "post"): (200, {**job.model_dump(), "status": "cancelled", "cancel_requested": True}, "application/json"),
        ("/api/v1/jobs/{job_id}/results", "get"): (200, {"artifacts": [{"id": "0", "name": "synthetic_0001.pdf", "media_type": "application/pdf", "size": 12000}, {"id": "2", "name": "dataset.zip", "media_type": "application/zip", "size": 9000}]}, "application/json"),
        ("/api/v1/jobs/{job_id}/shares", "get"): (200, [share.model_dump()], "application/json"),
        ("/api/v1/jobs/{job_id}/shares", "post"): (201, ShareCreated(**share.model_dump(), token="CHAVE_PRIVADA_DO_LINK").model_dump(), "application/json"),
        ("/api/v1/shares/{share_id}", "delete"): (200, {**share.model_dump(), "revoked_at": timestamp + 60}, "application/json"),
        ("/api/v1/shared/{share_id}", "get"): (200, {"share": share.model_dump(), "job": detail.model_dump(), "artifacts": []}, "application/json"),
    }
    names = {
        "health": "Saúde dos serviços", "create_workspace": "Criar espaço privado", "document_types": "Tipos de documento",
        "models": "Listar modelos", "create_model": "Cadastrar template PDF", "duplicate_model": "Criar cópia editável",
        "update_model": "Atualizar metadados", "delete_model": "Excluir modelo", "template": "Template PDF",
        "model_preview": "Prévia PNG do template", "mapping": "Mapeamento e campos", "update_mapping": "Salvar mapeamento",
        "capabilities": "Modelos, filtros e limites", "writing_presets": "Presets incluídos", "presets": "Listar presets privados",
        "create_preset": "Criar preset", "update_preset": "Atualizar preset", "delete_preset": "Excluir preset",
        "handwriting_preview": "Gerar amostras de escrita", "realism_preview": "Comparar realismo",
        "configurations": "Listar configurações", "create_configuration": "Salvar configuração",
        "delete_configuration": "Excluir configuração", "validate": "Validar geração", "submit": "Criar dataset",
        "jobs": "Histórico de gerações", "job": "Aguardar conclusão", "cancel": "Cancelar job", "results": "Listar artefatos",
        "artifact": "Baixar artefato", "shares": "Listar links", "create_share": "Criar link controlado",
        "revoke_share": "Revogar link", "shared": "Abrir resultado compartilhado", "shared_artifact": "Abrir artefato compartilhado",
    }
    bodies = {
        "create_workspace": {},
        "create_model": {"label": "Meu modelo", "description": "Template de pesquisa", "filename": "template.pdf", "pdf_base64": "{{pdfBase64}}", "mapping_id": "default", "mapping": None},
        "duplicate_model": {}, "update_model": {"label": "Meu modelo", "description": "Versão revisada", "status": "draft"},
        "update_mapping": {"definition": "__MAPPING__"},
        "create_preset": {"name": "Escrita azul natural", "writing": Writing().model_dump(), "filter_preset": "none", "shareable": False},
        "update_preset": {"name": "Escrita azul revisada", "writing": Writing().model_dump(), "filter_preset": "light", "shareable": False},
        "handwriting_preview": {"writing": Writing().model_dump(), "texts": ["Ana Silva Santos", "15092026", "SP"]},
        "realism_preview": {"document_type": "{{document_type}}", "preset": "medium", "seed": 42},
        "create_configuration": {"name": "Configuração DNV", "options": options.model_dump()},
        "validate": options.model_dump(), "submit": options.model_dump(), "cancel": {},
        "create_share": {"allow_download": True, "expires_in_hours": 24},
    }
    groups: dict[str, list] = {}
    for path, methods in schema["paths"].items():
        for method, operation in methods.items():
            function = operation["operationId"].split("_api_v1")[0]
            resolved = path
            for parameter in ("document_type", "mapping_id", "model_id", "preset_id", "configuration_id", "job_id", "artifact_id", "share_id"):
                resolved = resolved.replace("{" + parameter + "}", "{{" + parameter + "}}")
            query = [{"key": parameter["name"], "value": str(parameter["schema"].get("default", "")).lower()}
                     for parameter in operation.get("parameters", []) if parameter["in"] == "query"]
            raw = "{{baseUrl}}" + resolved + ("?" + "&".join(item["key"] + "=" + item["value"] for item in query) if query else "")
            headers = [{"key": "Accept", "value": "*/*"}]
            if any(parameter["name"] == "x-workspace-token" for parameter in operation.get("parameters", [])):
                headers.append({"key": "X-Workspace-Token", "value": "{{workspace_token}}"})
            auth = {"type": "noauth"}
            if operation.get("security") and path.startswith("/api/v1/shared"):
                auth = {"type": "bearer", "bearer": [{"key": "token", "value": "{{share_token}}", "type": "string"}]}
            elif operation.get("security") and not any(header["key"] == "X-Workspace-Token" for header in headers):
                auth = {"type": "bearer", "bearer": [{"key": "token", "value": "{{job_token}}", "type": "string"}]}
            request = {"method": method.upper(), "header": headers, "auth": auth,
                       "url": {"raw": raw, "host": ["{{baseUrl}}"], "path": resolved.lstrip("/").split("/"), "query": query},
                       "description": f"Contrato implementado: {method.upper()} {path}."}
            if method in {"post", "put", "patch"}:
                request["header"].append({"key": "Content-Type", "value": "application/json"})
                body = copy.deepcopy(bodies.get(function, {}))
                text = json.dumps(body, ensure_ascii=False, indent=2)
                text = text.replace('"__MAPPING__"', "{{mappingDefinition}}")
                request["body"] = {"mode": "raw", "raw": text, "options": {"raw": {"language": "json"}}}
            tests = ["pm.test('Resposta HTTP válida', () => pm.expect(pm.response.code).to.be.below(500));"]
            if function == "create_workspace": tests += ["const value = pm.response.json();", "pm.collectionVariables.set('workspace_token', value.token);"]
            if function == "document_types": tests += ["const value = pm.response.json()[0];", "pm.collectionVariables.set('document_type', value.id);", "pm.collectionVariables.set('mapping_id', value.mappings[0].id);", "pm.collectionVariables.set('model_id', value.id);"]
            if function == "mapping": tests += ["const value = pm.response.json();", "pm.collectionVariables.set('fields', JSON.stringify(value.fields.slice(0,3).map(item => item.field_id)));", "pm.collectionVariables.set('mappingDefinition', JSON.stringify(value.definition));"]
            if function == "create_preset": tests += ["pm.collectionVariables.set('preset_id', pm.response.json().id);"]
            if function == "create_configuration": tests += ["pm.collectionVariables.set('configuration_id', pm.response.json().id);"]
            if function == "submit": tests += ["const value = pm.response.json();", "pm.collectionVariables.set('job_id', value.id);", "pm.collectionVariables.set('job_token', value.token);", "pm.collectionVariables.set('poll_count', 0);"]
            if function == "job": tests += ["const value = pm.response.json();", "const count = Number(pm.collectionVariables.get('poll_count') || 0) + 1;", "pm.collectionVariables.set('poll_count', count);", "if (['queued','running'].includes(value.status) && count < 600) { setTimeout(() => {}, 1500); pm.execution.setNextRequest(pm.info.requestName); }"]
            if function == "results": tests += ["const files = pm.response.json().artifacts;", "pm.collectionVariables.set('artifact_id', files.find(item => item.media_type === 'application/zip').id);"]
            if function == "create_share": tests += ["const value = pm.response.json();", "pm.collectionVariables.set('share_id', value.id);", "pm.collectionVariables.set('share_token', value.token);"]
            response_items = []
            example = examples.get((path, method))
            if example:
                code, body, mime = example
                response_items.append({"name": "Resposta de exemplo", "originalRequest": copy.deepcopy(request), "status": "OK", "code": code,
                                       "header": [{"key": "Content-Type", "value": mime}],
                                       "body": json.dumps(body, ensure_ascii=False, indent=2) if mime == "application/json" else ""})
            elif function in {"template", "model_preview", "artifact", "shared_artifact"}:
                mime = "application/pdf" if function in {"template", "artifact", "shared_artifact"} else "image/png"
                response_items.append({"name": "Arquivo binário", "originalRequest": copy.deepcopy(request), "status": "OK", "code": 200,
                                       "header": [{"key": "Content-Type", "value": mime}], "body": ""})
            groups.setdefault(operation["tags"][0], []).append({"name": names[function], "request": request, "response": response_items, "event": event(tests)})
    variables = {
        "baseUrl": "http://127.0.0.1:8000", "workspace_token": "", "document_type": document["id"], "mapping_id": mapping_id,
        "model_id": document["id"], "preset_id": "", "configuration_id": "", "fields": json.dumps(options.fields),
        "mappingDefinition": json.dumps(mapping["definition"], ensure_ascii=False), "pdfBase64": "COLE_BASE64_DO_PDF",
        "job_id": "", "job_token": "", "artifact_id": "", "share_id": "", "share_token": "", "poll_count": "0",
    }
    order = ["Operação", "Espaço de trabalho", "Catálogo", "Modelos", "Presets", "Configurações", "Geração", "Jobs", "Resultados", "Compartilhamento"]
    items = [{"name": name, "item": groups.get(name, [])} for name in order]
    items.append({"name": "Contrato", "item": [{"name": "OpenAPI", "request": {"method": "GET", "auth": {"type": "noauth"}, "url": "{{baseUrl}}/api/openapi.json"}, "event": event(["pm.test('HTTP 200', () => pm.response.to.have.status(200));"])}, {"name": "Documentação interativa", "request": {"method": "GET", "auth": {"type": "noauth"}, "url": "{{baseUrl}}/api/docs"}}]})
    return {"info": {"name": "OCRSD API v1", "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json",
                     "description": "Fluxo real: criar espaço privado, consultar catálogo, validar, criar job, acompanhar, listar artefatos, baixar e compartilhar. Execute na ordem. Para upload, preencha pdfBase64."},
            "variable": [{"key": key, "value": value, "type": "string"} for key, value in variables.items()], "item": items}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    path = ROOT / "postman/ocrsd.postman_collection.json"
    collection = build_collection()
    if args.check:
        if json.loads(path.read_text(encoding="utf-8")) != collection:
            raise SystemExit("Coleção desatualizada; execute python -m scripts.export_postman")
    else:
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(collection, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
