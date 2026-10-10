from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi.testclient import TestClient

from src.web.api import create_app
from src.web.models import GenerationOptions, JobResponse, SubmittedJob
from src.web.settings import Settings

ROOT = Path(__file__).resolve().parents[1]


def event(lines: list[str]) -> list[dict]:
    return [{"listen": "test", "script": {"type": "text/javascript", "exec": lines}}]


def build_collection() -> dict:
    with TemporaryDirectory() as temporary:
        defaults = {name: field.default for name, field in Settings.model_fields.items()}
        app = create_app(Settings(**{**defaults, "storage": Path(temporary), "neural_enabled": False}))
        schema = app.openapi()
        with TestClient(app) as client:
            types = client.get("/api/v1/document-types").json()
            document = types[0]
            mapping_id = document["mappings"][0]["id"]
            mapping = client.get(f'/api/v1/document-types/{document["id"]}/mappings/{mapping_id}').json()
            options = GenerationOptions(document_type=document["id"], mapping_id=mapping_id,
                                        fields=[field["field_id"] for field in mapping["fields"][:3]])
            examples = {path: client.get(path).json() for path in (
                "/api/v1/health", "/api/v1/document-types", "/api/v1/capabilities", "/api/v1/writing-presets")}
    job = JobResponse(id="0" * 32, status="queued", completed=0, total=1,
                      created_at=1791648000, updated_at=1791648000, expires_at=1791734400,
                      cancel_requested=False, error=None).model_dump()
    examples.update({
        "/api/v1/document-types/{document_type}/mappings/{mapping_id}": mapping,
        "/api/v1/generation/validate": options.model_dump(),
        "/api/v1/jobs": SubmittedJob(**job, token="CHAVE_RETORNADA_APENAS_NA_CRIACAO").model_dump(),
        "/api/v1/jobs/{job_id}": job,
        "/api/v1/jobs/{job_id}/cancel": {**job, "status": "cancelled", "cancel_requested": True},
        "/api/v1/jobs/{job_id}/results": {"artifacts": [
            {"id": "0", "name": "synthetic_0001.pdf", "media_type": "application/pdf", "size": 12000},
            {"id": "1", "name": "dataset.zip", "media_type": "application/zip", "size": 9000}]},
    })
    names = {
        "health": "Saúde dos serviços", "document_types": "Tipos de documento", "template": "Template PDF",
        "mapping": "Mapeamento e campos", "capabilities": "Modelos, filtros e limites", "writing_presets": "Presets de escrita",
        "validate": "Validar configuração", "submit": "Criar dataset", "job": "Aguardar conclusão", "cancel": "Cancelar job",
        "results": "Listar resultados", "artifact": "Baixar dataset ZIP",
    }
    groups: dict[str, list] = {}
    for path, methods in schema["paths"].items():
        for method, operation in methods.items():
            function = operation["operationId"].split("_api_")[0]
            name = names[function]
            resolved = path
            for parameter in ("document_type", "mapping_id", "job_id", "artifact_id"):
                resolved = resolved.replace("{" + parameter + "}", "{{" + parameter + "}}")
            query = [{"key": parameter["name"], "value": str(parameter["schema"].get("default", "")).lower()}
                     for parameter in operation.get("parameters", []) if parameter["in"] == "query"]
            raw = "{{baseUrl}}" + resolved + ("?" + "&".join(item["key"] + "=" + item["value"] for item in query) if query else "")
            request = {"method": method.upper(), "header": [{"key": "Accept", "value": "*/*"}],
                       "auth": {"type": "bearer", "bearer": [{"key": "token", "value": "{{job_token}}", "type": "string"}]} if operation.get("security") else {"type": "noauth"},
                       "url": {"raw": raw, "host": ["{{baseUrl}}"], "path": resolved.lstrip("/").split("/"), "query": query},
                       "description": f"Contrato: {method.upper()} {path}. IDs obtidos no catálogo; nenhuma entrada aceita caminhos locais."}
            if method == "post":
                request["header"].append({"key": "Content-Type", "value": "application/json"})
                body = options.model_dump() if function in ("submit", "validate") else {}
                request["body"] = {"mode": "raw", "raw": json.dumps(body, ensure_ascii=False, indent=2), "options": {"raw": {"language": "json"}}}
                if body:
                    request["body"]["raw"] = request["body"]["raw"].replace(json.dumps(document["id"]), '"{{document_type}}"').replace(json.dumps(mapping_id), '"{{mapping_id}}"')
                    request["body"]["raw"] = request["body"]["raw"].replace(json.dumps(options.fields, indent=2).replace('\n', '\n  '), '{{fields}}')
            code = 202 if function == "submit" else 200
            tests = [f'pm.test("HTTP {code}", () => pm.response.to.have.status({code}));']
            if function == "document_types":
                tests += ['const type = pm.response.json()[0];', 'pm.collectionVariables.set("document_type", type.id);', 'pm.collectionVariables.set("mapping_id", type.mappings[0].id);']
            if function == "mapping":
                tests += ['pm.collectionVariables.set("fields", JSON.stringify(pm.response.json().fields.slice(0, 3).map(field => field.field_id)));']
            if function == "submit":
                tests += ['const job = pm.response.json();', 'pm.collectionVariables.set("job_id", job.id);', 'pm.collectionVariables.set("job_token", job.token);', 'pm.collectionVariables.set("poll_count", 0);']
            if function == "job":
                tests += ['const job = pm.response.json();', 'const attempts = Number(pm.collectionVariables.get("poll_count") || 0) + 1;', 'pm.collectionVariables.set("poll_count", attempts);',
                          'if (["queued", "running"].includes(job.status) && attempts < 600) {', '  pm.execution.setNextRequest(pm.info.requestName);', '  setTimeout(() => {}, 1500);',
                          '} else {', '  pm.test("Dataset concluído", () => pm.expect(job.status).to.eql("succeeded"));', '  if (job.status !== "succeeded") pm.execution.setNextRequest(null);', '}']
            if function == "results":
                tests += ['const files = pm.response.json().artifacts;', 'pm.collectionVariables.set("artifact_id", files.find(file => file.media_type === "application/zip").id);', 'pm.collectionVariables.set("pdf_id", files.find(file => file.media_type === "application/pdf").id);']
            mime = "application/json" if path in examples else "application/pdf" if function == "template" else "application/zip"
            response = {"name": "Resposta de exemplo", "originalRequest": copy.deepcopy(request), "status": "Accepted" if code == 202 else "OK", "code": code,
                        "header": [{"key": "Content-Type", "value": mime}], "body": json.dumps(examples[path], ensure_ascii=False, indent=2) if path in examples else ""}
            responses = [response]
            if operation.get("security"):
                unauthorized = copy.deepcopy(request)
                unauthorized["auth"] = {"type": "noauth"}
                responses.append({"name": "Chave ausente", "originalRequest": unauthorized, "status": "Unauthorized", "code": 401,
                                  "header": [{"key": "Content-Type", "value": "application/json"}],
                                  "body": json.dumps({"error": {"code": "unauthorized", "message": "Informe a chave de acesso do job"}}, ensure_ascii=False)})
            if function in ("submit", "validate"):
                invalid = copy.deepcopy(request)
                invalid["body"]["raw"] = json.dumps({**options.model_dump(), "count": 0}, ensure_ascii=False)
                responses.append({"name": "Quantidade inválida", "originalRequest": invalid, "status": "Unprocessable Entity", "code": 422,
                                  "header": [{"key": "Content-Type", "value": "application/json"}],
                                  "body": json.dumps({"error": {"code": "validation_error", "message": "Parâmetros inválidos; confira os campos e limites no contrato da API"}}, ensure_ascii=False)})
            item = {"name": name, "request": request, "response": responses, "event": event(tests)}
            group = "Cancelamento" if function == "cancel" else operation["tags"][0]
            groups.setdefault(group, []).append(item)
            if function == "artifact":
                preview = copy.deepcopy(item)
                preview["name"] = "Visualizar PDF"
                preview["request"]["url"]["raw"] = raw.replace("{{artifact_id}}", "{{pdf_id}}").replace("download=true", "download=false")
                preview["request"]["url"]["path"][-1] = "{{pdf_id}}"
                preview["request"]["url"]["query"][0]["value"] = "false"
                preview["response"][0]["header"][0]["value"] = "application/pdf"
                preview["response"][0]["originalRequest"] = copy.deepcopy(preview["request"])
                groups[group].append(preview)
    cancel_create = copy.deepcopy(next(item for item in groups["Geração"] if item["name"] == "Criar dataset"))
    cancel_create["name"] = "Criar job para cancelamento"
    groups["Cancelamento"].insert(0, cancel_create)
    groups["Contrato"] = [{"name": name, "request": {"method": "GET", "auth": {"type": "noauth"}, "url": "{{baseUrl}}" + path},
                           "event": event(['pm.test("HTTP 200", () => pm.response.to.have.status(200));'])}
                          for name, path in (("OpenAPI", "/api/openapi.json"), ("Documentação interativa", "/api/docs"))]
    variables = {"baseUrl": "http://127.0.0.1:5173", "document_type": document["id"], "mapping_id": mapping_id,
                 "fields": json.dumps(options.fields), "job_id": "", "job_token": "", "artifact_id": "", "pdf_id": "", "poll_count": "0"}
    return {"info": {"name": "OCRSD API v1", "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json",
                     "description": "Execute a coleção na ordem para descobrir recursos, validar, gerar, aguardar, visualizar e baixar. Cancelamento cria outro job. Configure baseUrl sem barra final. A chave do job é temporária e privada; não exporte variáveis preenchidas."},
            "variable": [{"key": key, "value": value, "type": "string"} for key, value in variables.items()],
            "item": [{"name": name, "item": groups[name]} for name in ("Operação", "Catálogo", "Geração", "Jobs", "Resultados", "Cancelamento", "Contrato")]}


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
