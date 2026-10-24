from __future__ import annotations

import json
import os
from pathlib import Path

CATALOG_FILENAME = "catalogo.json"
STATUSES = {"disponivel", "planejado"}
PRIORITIES = {"alta", "media", "baixa"}


def catalog_path(root: Path) -> Path:
    return root / CATALOG_FILENAME


def load_catalog(root: Path) -> dict:
    path = catalog_path(root)
    try:
        data = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    except (OSError, ValueError):
        data = {}
    if not isinstance(data, dict):
        data = {}
    orgaos = data.get("orgaos")
    documentos = data.get("documentos")
    return {"orgaos": orgaos if isinstance(orgaos, dict) else {},
            "documentos": documentos if isinstance(documentos, dict) else {}}


def save_catalog(root: Path, data: dict) -> None:
    path = catalog_path(root)
    payload = {"orgaos": data.get("orgaos") or {}, "documentos": data.get("documentos") or {}}
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _is_http_url(value: object) -> bool:
    return isinstance(value, str) and value.startswith(("http://", "https://"))


def catalog_issues(catalog: dict, discovered_types: dict) -> list[str]:
    issues: list[str] = []
    orgaos = catalog.get("orgaos") or {}
    documentos = catalog.get("documentos") or {}
    for key, orgao in orgaos.items():
        if not isinstance(orgao, dict):
            issues.append(f"Órgão {key}: definição inválida")
            continue
        site = orgao.get("site")
        if site is not None and not _is_http_url(site):
            issues.append(f"Órgão {key}: o site deve ser uma URL http(s)")
    for key, entry in documentos.items():
        if not isinstance(entry, dict):
            issues.append(f"Documento {key}: definição inválida")
            continue
        orgao = entry.get("orgao")
        if orgao not in orgaos:
            issues.append(f"Documento {key}: órgão desconhecido “{orgao}”")
        status = entry.get("status")
        if status not in STATUSES:
            issues.append(f"Documento {key}: status inválido “{status}”")
        prioridade = entry.get("prioridade")
        if prioridade not in PRIORITIES:
            issues.append(f"Documento {key}: prioridade inválida “{prioridade}”")
        documentacao = entry.get("documentacao")
        if documentacao is not None and not _is_http_url(documentacao):
            issues.append(f"Documento {key}: documentacao deve ser uma URL http(s)")
        variante_de = entry.get("variante_de")
        if variante_de is not None and variante_de not in documentos:
            issues.append(f"Documento {key}: variante_de aponta para um documento inexistente “{variante_de}”")
        if status == "disponivel" and key not in discovered_types and not variante_de:
            issues.append(f"Documento {key}: marcado como disponível, mas não há pasta em documents/{key}")
    for key in discovered_types:
        if key not in documentos:
            issues.append(f"A pasta documents/{key} não possui entrada no catálogo")
    return issues
