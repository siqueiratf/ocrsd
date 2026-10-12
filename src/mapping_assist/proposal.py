from __future__ import annotations

from datetime import datetime, timezone
import re
from typing import TYPE_CHECKING, Any

from src.mapping_assist.matching import Match, guess_data_type, section_of
from src.semantic import VERSION
from src.semantic.custom import validate_custom_definition
from src.semantic.plan import semantic_plan
from src.semantic.registry import REGISTRY
from src.semantic.resolve import document_context, normalize

if TYPE_CHECKING:
    from src.mapping_assist.analysis import FieldCandidate, PageAnalysis

DPI = 300
TEXT_TYPES = {"name": "person_name", "uf": "uf", "municipality": "municipality", "date": "date", "time": "time", "sex": "sex",
              "cep": "cep", "cns": "cns", "cpf": "document", "dnv_number": "dnv_number", "integer": "integer",
              "occupation": "occupation"}
VALIDATION = {"date": {"format": "DD/MM/AAAA"}, "document": {"document_kind": "cpf"}}
DATE_WORDS = {"dia": "day", "mes": "month", "ano": "year"}


def _sex_value(label: str, code: str) -> str | None:
    text = normalize(label)
    code = normalize(code)
    for value, prefix in (("M", "masc"), ("F", "fem")):
        if code == value.lower() or re.match(rf"^(?:{value.lower()}\b|{prefix}[a-z]*\b)", text):
            return value
    return None


def _slug(text: str) -> str:
    return "_".join(normalize(text).split()[:6])[:60]


def _box(box: Any, scale: float, size: tuple[int, int]) -> list[int] | None:
    try:
        x1, y1, x2, y2 = (round(float(value) * scale) for value in box)
    except (TypeError, ValueError):
        return None
    x1, x2 = max(0, min(x1, x2)), min(size[0], max(x1, x2))
    y1, y2 = max(0, min(y1, y2)), min(size[1], max(y1, y2))
    return [x1, y1, x2, y2] if x2 > x1 and y2 > y1 else None


def _date_part(label: str, key: str | None) -> str | None:
    words = set(normalize(label).split())
    found = [part for word, part in DATE_WORDS.items() if word in words]
    return found[0] if key in REGISTRY and REGISTRY[key].kind == "date" and len(found) == 1 and "data" not in words else None


def _data_type(candidate: FieldCandidate, key: str | None, proposal: dict | None, custom: dict[str, dict]) -> str:
    if candidate.kind in {"checkbox", "digit_sequence"}:
        return candidate.kind
    if key:
        if key in REGISTRY:
            return TEXT_TYPES.get(REGISTRY[key].kind, "text")
        return "text" if custom[key]["data_type"] == "boolean" else custom[key]["data_type"]
    return (proposal or {}).get("data_type") or guess_data_type(candidate)


def _elements(candidate: FieldCandidate, field_id: str, scale: float, size: tuple[int, int], key: str | None = None) -> list[dict]:
    elements, used = [], set()
    if candidate.kind == "digit_sequence":
        ordered = sorted(candidate.elements, key=lambda item: (item.get("index") is None, item.get("index") or 0))
        for position, element in enumerate(ordered):
            if box := _box(element.get("bbox"), scale, size):
                elements.append({"element_id": f"{field_id}_{position}", "index": position,
                                 "label": f"{candidate.label} {position}", "bbox": box})
    elif candidate.kind == "checkbox":
        for position, element in enumerate(candidate.elements):
            label = str(element.get("label") or element.get("code") or f"Opção {position + 1}").strip()
            value = str(element.get("code") or label).strip()
            if key in REGISTRY and REGISTRY[key].kind == "sex":
                value = _sex_value(label, value) or value
            identifier = f"{field_id}_{_slug(value) or position}"
            identifier = identifier if identifier not in used else f"{identifier}_{position}"
            if box := _box(element.get("bbox"), scale, size):
                used.add(identifier)
                elements.append({"element_id": identifier, "label": label, "value": value, "bbox": box})
    return elements


def _custom_catalog(definitions: list[dict] | dict[str, dict] | None) -> dict[str, dict]:
    raw_custom = definitions.values() if isinstance(definitions, dict) else definitions or []
    custom = {}
    for spec in raw_custom:
        if spec.get("status", "approved") != "approved":
            continue
        normalized = validate_custom_definition(*(spec[name] for name in ("key", "label", "description", "data_type", "generator")))
        custom[normalized["key"]] = normalized
    return custom


def build_proposal(analysis: PageAnalysis, matches: list[Match], document: str | None, provenance: dict | None = None,
                   *, custom_definitions: list[dict] | dict[str, dict] | None = None) -> dict:
    provenance = dict(provenance or {})
    custom = _custom_catalog(custom_definitions)
    context = document_context(document)
    template = str(provenance.pop("template_id", None) or document or "modelo_web")
    scale = DPI / analysis.dpi if analysis.dpi else 1.0
    size = (round(analysis.page_size_px[0] * scale), round(analysis.page_size_px[1] * scale))
    by_id = {match.candidate_id: match for match in matches}
    warnings, fields, identifiers = list(analysis.warnings), [], set()
    for candidate in analysis.candidates:
        match = by_id.get(candidate.id) or Match(candidate.id, None, "new", "none", None, (), "Candidato sem correspondência calculada.")
        box = _box(candidate.region, scale, size)
        if box is None:
            warnings.append(f"Candidato {candidate.id} ignorado: área fora da página")
            continue
        key = match.key if match.status == "matched" and match.key in REGISTRY | custom else None
        base = key.replace(".", "_") if key else _slug(candidate.label) or f"campo_{_slug(candidate.id) or len(fields) + 1}"
        field_id, number = base, 2
        while field_id in identifiers:
            field_id, number = f"{base}_{number}", number + 1
        identifiers.add(field_id)
        label = candidate.label.strip() or field_id
        part = _date_part(label, key)
        data_type = "integer" if part and candidate.kind == "text" else _data_type(candidate, key, match.proposal, custom)
        issues = []
        if data_type == "checkbox" and len(candidate.elements) < 2:
            data_type = "text"
            issues.append("opção de seleção isolada; associe-a ao campo correspondente ou rejeite-a")
            key = None
        field = {"field_id": field_id, "label": label, "description": label, "data_type": data_type, "bbox": box}
        elements = _elements(candidate, field_id, scale, size, key) if data_type in {"checkbox", "digit_sequence"} else []
        if data_type == "checkbox" and len(elements) < 2:
            data_type = field["data_type"] = "text"
            elements = []
            key = None
            if not issues:
                issues.append("opção de seleção isolada; associe-a ao campo correspondente ou rejeite-a")
        if elements:
            field["elements"] = elements
        if validation := VALIDATION.get(field["data_type"]):
            field["validation"] = dict(validation)
        if bloco := section_of(context, candidate.section, candidate.section_title) or candidate.section:
            field["bloco"] = bloco
        if field["data_type"] == "digit_sequence" and not elements:
            field["data_type"] = "text"
            issues.append("sequência sem caixas detectadas; tratada como texto")
        status, reason = match.status, match.reason
        if candidate.kind == "checkbox" and data_type == "text":
            status, reason = "ambiguous", f"{reason} Opção de seleção isolada sem campo composto válido."
        if key:
            field["semantic"] = {"key": key, "part": part} if part else key
            if key in REGISTRY:
                decision = semantic_plan([field], template).decisions[field_id]
                if decision.issue:
                    del field["semantic"]
                    status, reason = "ambiguous", f"{reason} Vínculo não aplicado: {decision.issue}."
                    issues.append(decision.issue)
        field["suggestion"] = {
            "status": "suggested", "candidate_id": candidate.id, "match": status, "method": match.method, "score": match.score,
            "alternatives": [{"key": alternative, "label": REGISTRY[alternative].label if alternative in REGISTRY else custom[alternative]["label"],
                              "score": score} for alternative, score in match.alternatives if alternative in REGISTRY or alternative in custom],
            "reason": reason, "proposal": dict(match.proposal) if match.proposal else None,
            **({"issues": issues} if issues else {})}
        warnings.extend(f"{label}: {issue}" for issue in issues)
        fields.append(field)
    meta = {"method": "ocrsd-hybrid", "analysis_source": analysis.source, "provider": "none", "model": None,
            "registry_version": VERSION, "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "reviewed": False, "page": analysis.page, "analysis_dpi": analysis.dpi}
    meta.update({name: value for name, value in provenance.items() if name not in {"warnings", "timings"}})
    meta["warnings"] = warnings + [str(item) for item in provenance.get("warnings") or []]
    meta["timings"] = {**dict(analysis.timings or {}), **dict(provenance.get("timings") or {})}
    used_custom = {key: custom[key] for field in fields
                   if isinstance(key := field.get("semantic"), str) and key in custom}
    return {"template_id": template, "dpi": DPI, "page_size_px": list(size), "fields": fields,
            **({"semantic_definitions": used_custom} if used_custom else {}), "provenance": meta}


def accept_suggestion(field: dict, key: str | None = None,
                      *, custom_definitions: list[dict] | dict[str, dict] | None = None) -> dict:
    accepted = {name: value for name, value in field.items() if name != "suggestion"}
    if key is not None:
        if key not in REGISTRY and key not in _custom_catalog(custom_definitions):
            raise ValueError(f"Chave semântica desconhecida: {key}")
        accepted["semantic"] = key
    return accepted


def strip_suggestions(definition: dict) -> dict:
    fields = definition.get("fields") or []
    pending = [field for field in fields if isinstance(field.get("suggestion"), dict)
               and field["suggestion"].get("status", "suggested") == "suggested"]
    if pending:
        raise ValueError(f"Há {len(pending)} sugestão(ões) pendente(s) de revisão.")
    stripped = {**definition, "fields": [{name: value for name, value in field.items() if name != "suggestion"} for field in fields]}
    if isinstance(definition.get("provenance"), dict):
        stripped["provenance"] = {**definition["provenance"], "reviewed": True,
                                  "reviewed_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    return stripped
