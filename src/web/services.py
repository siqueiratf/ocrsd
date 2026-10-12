from __future__ import annotations

import base64
import functools
import hashlib
import io
import json
from pathlib import Path
import random
import sqlite3
import tempfile
import threading
from typing import Callable
from zipfile import ZIP_DEFLATED, ZipFile

from src.document_types import discover_document_types
from src.field_rules import COMPOSITE_TYPES, DATE_PARTS, SEPARATED_DATE, apply_shared_rules, rule_errors
from src.semantic.plan import semantic_plan
from src.web.models import DocumentModelResponse, GenerationOptions, PresetResponse, Writing
from src.web.settings import Settings
from src.writing_presets import list_presets

NOTICE = "DOCUMENTO SINTÉTICO — SEM VALIDADE LEGAL"
STANDARD_PRESETS = (("padrao_fonte_natural", "Fonte manuscrita · natural", 0.65),
                    ("padrao_fonte_regular", "Fonte manuscrita · regular", 0.25),
                    ("padrao_fonte_variada", "Fonte manuscrita · variada", 0.95))


class ServiceError(Exception):
    def __init__(self, status: int, code: str, message: str):
        self.status, self.code, self.message = status, code, message
        super().__init__(message)


class Catalog:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.types = discover_document_types(settings.documents)

    def custom(self, key: str, owner_id: str | None = None) -> dict:
        database = self.settings.storage.resolve() / "jobs.sqlite3"
        if not database.is_file():
            raise ServiceError(404, "not_found", "Tipo de documento não encontrado")
        with sqlite3.connect(database) as connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute("SELECT * FROM models WHERE id=?", (key,)).fetchone()
        if not row or owner_id is None or row["owner_id"] != owner_id:
            raise ServiceError(404, "not_found", "Tipo de documento não encontrado")
        return dict(row)

    def document(self, key: str, owner_id: str | None = None):
        if key in self.types:
            return self.types[key]
        from src.document_types import DocumentType

        model = self.custom(key, owner_id)
        root = self.settings.storage.resolve() / "models" / key
        return DocumentType(key=key, root=root, template=root / "template.pdf",
                            mappings=(root / f'{model["mapping_id"]}.json',), presets=(), dataset=root / "dataset")

    def mapping(self, key: str, mapping_id: str, owner_id: str | None = None) -> Path:
        for path in self.document(key, owner_id).mappings:
            if path.stem == mapping_id:
                return path
        raise ServiceError(404, "not_found", "Mapeamento não encontrado")

    def definition(self, key: str, mapping_id: str, owner_id: str | None = None) -> dict:
        definition = json.loads(self.mapping(key, mapping_id, owner_id).read_text(encoding="utf-8"))
        definition["source_template"] = key
        return definition

    def source(self, options: GenerationOptions, owner_id: str | None = None) -> dict:
        return {"definition": self.definition(options.document_type, options.mapping_id, owner_id),
                "template": self.document(options.document_type, owner_id).template}

    def active_mapping(self, key: str, owner_id: str | None = None) -> str:
        if key in self.types:
            return self.types[key].active_mapping.stem
        return self.custom(key, owner_id)["mapping_id"]

    def models(self) -> list[dict]:
        assets = Path(__file__).resolve().parents[2] / "assets" / "handwriting"
        paths = {"sjvasquez": assets / "sjvasquez/weights.npz", "aaron1259": assets / "weights.bin", "wpvnteam": assets / "weights.bin"}
        available = {name: self.settings.neural_enabled and path.is_file() for name, path in paths.items()}
        available["sjvasquez"] = available["sjvasquez"] and all(
            (assets / "sjvasquez" / relative).is_file()
            for relative in ["weights.sha256"] + [f"styles/style-{index}-{kind}.npy" for index in range(13) for kind in ("chars", "strokes")]
        )
        return [{"id": "font", "label": "Fonte manuscrita (local)", "available": True, "reason": None}] + [
            {"id": name, "label": name, "available": available[name],
             "reason": None if available[name] else "Requer habilitação administrativa e pesos locais completos"}
            for name, path in paths.items()
        ]

    def presets(self) -> list[PresetResponse]:
        result = [PresetResponse(id=identifier, label=label, writing=Writing(intensity=intensity), filter_preset="none", available=True)
                  for identifier, label, intensity in STANDARD_PRESETS]
        available = {m["id"] for m in self.models() if m["available"]}
        for preset in list_presets(self.settings.documents / "writing_presets.json"):
            settings = preset.get("settings", {})
            writing = None
            reason = "Preset misto disponível apenas no desktop"
            if settings.get("mode") != "mixed":
                try:
                    writer = settings.get("writer")
                    writing = Writing(
                        model=settings.get("model", "font"),
                        writer=None if writer in (None, "", "Aleatório") else int(writer),
                        bias=settings.get("bias", 0.75), color=settings.get("pen_color", "#193787"),
                        width=settings.get("pen_width", 1.5), pressure=settings.get("pen_pressure", 0.25),
                        dropout=settings.get("pen_dropout", 0), text_policy=settings.get("text_policy", "normalize"),
                        handwriting=settings.get("handwriting", True), text_handwriting=settings.get("text_handwriting", True),
                        intensity=settings.get("handwriting_intensity", 0.65),
                    )
                    reason = None if writing.model in available else "Modelo indisponível neste servidor"
                except (ValueError, TypeError):
                    reason = "Preset incompatível com a versão web"
            result.append(PresetResponse(id=preset["uid"], label=preset.get("name", preset["uid"]), writing=writing,
                                         filter_preset=settings.get("filter_preset", "medium") if settings.get("realistic_filters") else "none",
                                         available=reason is None, reason=reason))
        return result

    def require_model(self, model: str) -> None:
        if not any(item["id"] == model and item["available"] for item in self.models()):
            raise ServiceError(422, "model_unavailable", "Modelo de escrita indisponível")

    def document_types(self, custom: list[dict]) -> list[dict]:
        result = [{"id": item.key, "label": item.label, "builtin": True, "active_mapping": item.active_mapping.stem,
                   "mappings": [{"id": path.stem, "label": path.stem.replace("_", " ")}
                                for path in sorted(item.mappings, key=lambda path: path != item.active_mapping)]}
                  for item in self.types.values()]
        result.extend({"id": row["id"], "label": row["label"], "builtin": False, "active_mapping": row["mapping_id"],
                       "mappings": [{"id": row["mapping_id"], "label": row["mapping_id"].replace("_", " ")}]}
                      for row in custom)
        return result

    def document_models(self, custom: list[dict]) -> list[DocumentModelResponse]:
        builtins = []
        for item in self.types.values():
            definition = json.loads(item.active_mapping.read_text(encoding="utf-8"))
            builtins.append(DocumentModelResponse(id=item.key, label=item.label, description="Modelo padrão do OCRSD",
                                                  status="published", version=1, field_count=len(definition["fields"]),
                                                  mapping_id=item.active_mapping.stem, builtin=True, editable=False))
        return builtins + [custom_model(row) for row in custom]

    def validate(self, options: GenerationOptions, owner_id: str | None = None, definition: dict | None = None) -> GenerationOptions:
        if options.mapping_id is None:
            options = options.model_copy(update={"mapping_id": self.active_mapping(options.document_type, owner_id)})
        definition = validate_mapping(definition or self.definition(options.document_type, options.mapping_id, owner_id))
        fields = {str(field["field_id"]) for field in definition["fields"]}
        if set(options.fields) - fields or len(set(options.fields)) != len(options.fields):
            raise ServiceError(422, "invalid_fields", "Selecione campos existentes, sem repetições")
        if options.count > self.settings.max_count:
            raise ServiceError(422, "quota", f"Limite de {self.settings.max_count} documentos por job")
        if not any(m["id"] == options.writing.model and m["available"] for m in self.models()):
            raise ServiceError(422, "model_unavailable", "Modelo não habilitado ou pesos ausentes")
        selected = [field for field in definition["fields"] if field["field_id"] in options.fields]
        issues = mapping_issues({**definition, "fields": selected})
        if issues:
            raise ServiceError(422, "invalid_mapping", issues[0])
        if options.writing.model != "font" and (options.writing.handwriting or options.writing.text_handwriting):
            from src.neural_handwriting import NeuralPen
            try:
                NeuralPen(**options.writing.model_dump(exclude={"handwriting", "text_handwriting", "intensity"}))
            except ValueError as exc:
                raise ServiceError(422, "writing_incompatible", f"Configuração de escrita inválida: {exc}") from exc
            conflicts = writing_issues(definition, options.fields, options.writing)
            if conflicts:
                raise ServiceError(422, "writing_incompatible", writing_message(options.writing, conflicts))
        return options


WRITING_SAMPLES = 64


def writing_issues(definition: dict, field_ids, writing: Writing) -> list[dict]:
    if writing.model == "font" or not (writing.handwriting or writing.text_handwriting):
        return []
    wanted = {str(item) for item in field_ids}
    fields = json.dumps([field for field in definition.get("fields", []) if str(field["field_id"]) in wanted], sort_keys=True)
    shared = json.dumps(definition.get("shared_rules") or {}, sort_keys=True)
    return [dict(item) for item in sampled_writing_issues(fields, shared, writing.text_policy, writing.handwriting, writing.text_handwriting,
                                                          str(definition.get("template_id") or ""))]


@functools.lru_cache(maxsize=256)
def sampled_writing_issues(fields: str, shared: str, policy: str, numbers: bool, texts: bool, document: str = "") -> tuple:
    from src.neural_handwriting import apply_text_policy, unsupported_characters
    from src.synthetic_dataset import field_texts, is_numeric_text
    issues = []
    for field in apply_shared_rules({"fields": json.loads(fields), "shared_rules": json.loads(shared)})["fields"]:
        if field.get("render_as_printed"):
            continue
        characters, too_long = set(), False
        for sample in range(WRITING_SAMPLES):
            for value in field_texts(field, random.Random(sample), {"document": document or None}):
                if not value.strip() or not (numbers if is_numeric_text(value) else texts):
                    continue
                text = apply_text_policy(value, policy)
                characters.update(unsupported_characters(text))
                too_long = too_long or len(text) > 160
        if characters or too_long:
            issues.append((("field_id", str(field["field_id"])),
                           ("label", field.get("label") or field.get("description") or str(field["field_id"])),
                           ("characters", tuple(sorted(characters))), ("too_long", too_long)))
    return tuple(issues)


def writing_message(writing: Writing, issues: list[dict]) -> str:
    characters = sorted({character for item in issues for character in item["characters"]})
    labels = ", ".join(item["label"] for item in issues[:4]) + (f" e mais {len(issues) - 4}" if len(issues) > 4 else "")
    if characters:
        shown = " ".join("espaço" if character == " " else character for character in characters[:12])
        advice = ("Use “Normalizar” em Caracteres não suportados, a fonte manuscrita ou remova esses campos"
                  if writing.text_policy == "strict" else "Use a fonte manuscrita ou remova esses campos")
        return f"O modelo {writing.model} não escreve os caracteres {shown} usados em {labels}. {advice}."
    return f"O modelo {writing.model} aceita até 160 caracteres por campo; {labels} excede esse limite. Use a fonte manuscrita ou remova esses campos."


def custom_model(row: dict) -> DocumentModelResponse:
    return DocumentModelResponse(id=row["id"], label=row["label"], description=row["description"],
                                 status=row["status"], version=row["version"], field_count=row["field_count"],
                                 mapping_id=row["mapping_id"], builtin=False, editable=True,
                                 created_at=row["created_at"], updated_at=row["updated_at"])


def create_model(store, settings: Settings, owner_id: str, label: str, description: str, mapping_id: str,
                 filename: str, pdf: bytes, mapping: dict | None = None) -> dict:
    if not filename.lower().endswith(".pdf"):
        raise ServiceError(422, "invalid_pdf", "Envie um arquivo PDF")
    if len(pdf) > settings.max_upload_bytes:
        raise ServiceError(413, "too_large", "O PDF excede o limite de upload")
    page_size = template_page_size(pdf)
    definition = validate_mapping(mapping or {"template_id": "modelo_web", "dpi": 300, "page_size_px": list(page_size), "fields": []}, page_size)
    return store.create_model(owner_id, label, description, mapping_id, pdf, definition)


def create_blank_model(store, owner_id: str, label: str, description: str, size: str, orientation: str) -> dict:
    page_size = page_dimensions(size, orientation)
    layout = {"page": {"size": size, "orientation": orientation}, "elements": []}
    definition = {"template_id": "modelo_composto", "dpi": 300, "page_size_px": list(page_size), "fields": []}
    return store.create_model(owner_id, label, description, "default", render_layout_pdf(layout, page_size), definition, layout)


def duplicate_model(catalog: Catalog, store, model_id: str, owner_id: str) -> dict:
    if model_id in catalog.types:
        document = catalog.document(model_id)
        mapping_id = document.active_mapping.stem
        definition = catalog.definition(model_id, mapping_id)
        label = document.label + " — cópia"
    else:
        source = store.model(model_id, owner_id)
        document = catalog.document(model_id, owner_id)
        mapping_id = source["mapping_id"]
        definition = catalog.definition(model_id, mapping_id, owner_id)
        label = source["label"] + " — cópia"
    definition["source_template"] = "template.pdf"
    layout = None if model_id in catalog.types else store.layout(model_id)
    return store.create_model(owner_id, label, "Cópia editável", mapping_id,
                              document.template.read_bytes(), validate_mapping(definition), layout)


BUILTIN_PROTECTED = "Modelos padrão são protegidos. Use “Duplicar e editar” para criar uma cópia editável."


def save_mapping(catalog: Catalog, store, model_id: str, owner_id: str, definition: dict) -> str:
    if model_id in catalog.types:
        raise ServiceError(403, "builtin_protected", BUILTIN_PROTECTED)
    model = store.model(model_id, owner_id)
    store.update_mapping(model_id, owner_id, validate_mapping(definition))
    return model["mapping_id"]


def mark_pdf(path: Path) -> None:
    import fitz

    with fitz.open(path) as document:
        for page in document:
            for top in (8, page.rect.height / 2, page.rect.height - 30):
                rect = fitz.Rect(8, top, page.rect.width - 8, top + 23)
                page.draw_rect(rect, color=(0.7, 0.05, 0.05), fill=(1, 1, 1), fill_opacity=0.88, overlay=True)
                page.insert_textbox(rect + (0, 4, 0, 0), NOTICE.replace("—", "-"), fontsize=11,
                                    fontname="hebo", color=(0.7, 0.05, 0.05), align=1, overlay=True)
        document.set_metadata({"title": NOTICE, "subject": "Uso educacional e de pesquisa"})
        with tempfile.NamedTemporaryFile(suffix=".pdf", dir=path.parent, delete=False) as temporary:
            temporary_path = Path(temporary.name)
        try:
            document.save(temporary_path, garbage=4, deflate=True, deflate_images=True, deflate_fonts=True, use_objstms=1)
            temporary_path.replace(path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()


def validate_mapping(definition: dict, page_size: tuple[int, int] | None = None) -> dict:
    if not isinstance(definition, dict) or not isinstance(definition.get("fields"), list):
        raise ServiceError(422, "invalid_mapping", "O mapeamento precisa conter uma lista de campos")
    try:
        dpi = int(definition.get("dpi", 300))
        size = tuple(int(value) for value in (definition.get("page_size_px") or page_size or ()))
    except (TypeError, ValueError):
        raise ServiceError(422, "invalid_mapping", "DPI ou dimensões da página inválidos")
    if dpi < 72 or dpi > 600 or len(size) != 2 or min(size) < 100 or max(size) > 20000:
        raise ServiceError(422, "invalid_mapping", "DPI ou dimensões da página fora dos limites")
    identifiers: set[str] = set()

    def bbox(value: object) -> list[int]:
        if not isinstance(value, list) or len(value) != 4 or any(type(item) is not int for item in value):
            raise ServiceError(422, "invalid_mapping", "Toda área deve ter quatro coordenadas inteiras")
        x1, y1, x2, y2 = value
        if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1 or x2 > size[0] or y2 > size[1]:
            raise ServiceError(422, "invalid_mapping", "Uma área está fora dos limites da página")
        return value

    fields = []
    for raw in definition["fields"]:
        if not isinstance(raw, dict):
            raise ServiceError(422, "invalid_mapping", "Definição de campo inválida")
        identifier = str(raw.get("field_id", "")).strip()
        if not identifier or len(identifier) > 100 or identifier in identifiers:
            raise ServiceError(422, "invalid_mapping", "IDs de campos devem ser únicos e não vazios")
        identifiers.add(identifier)
        errors = rule_errors(str(raw.get("data_type") or "text"), raw.get("validation")) or semantic_errors(raw, definition.get("template_id"))
        if errors:
            raise ServiceError(422, "invalid_mapping", f"Campo {identifier}: {errors[0]}")
        field = {**raw, "field_id": identifier, "label": str(raw.get("label") or identifier), "bbox": bbox(raw.get("bbox"))}
        elements = []
        element_ids: set[str] = set()
        if not isinstance(raw.get("elements", []), list):
            raise ServiceError(422, "invalid_mapping", "Subelementos precisam ser uma lista")
        for raw_element in raw.get("elements", []):
            if not isinstance(raw_element, dict):
                raise ServiceError(422, "invalid_mapping", "Definição de subelemento inválida")
            element_id = str(raw_element.get("element_id", "")).strip()
            if not element_id or element_id in element_ids:
                raise ServiceError(422, "invalid_mapping", "IDs de subelementos devem ser únicos e não vazios")
            element_ids.add(element_id)
            elements.append({**raw_element, "element_id": element_id, "bbox": bbox(raw_element.get("bbox"))})
        if elements:
            field["elements"] = elements
        fields.append(field)
    return {**definition, "template_id": str(definition.get("template_id") or "modelo_web"), "dpi": dpi,
            "page_size_px": list(size), "fields": fields}


PAGE_SIZES = {"a4": ("A4", 210.0, 297.0), "oficio": ("Ofício", 216.0, 330.0), "carta": ("Carta", 215.9, 279.4)}
ORIENTATIONS = {"portrait": "Retrato", "landscape": "Paisagem"}
LAYOUT_KINDS = {"text", "line", "box"}
LAYOUT_ALIGNMENTS = {"left", "center", "right"}


def page_dimensions(size: str, orientation: str, dpi: int = 300) -> tuple[int, int]:
    if size not in PAGE_SIZES or orientation not in ORIENTATIONS:
        raise ServiceError(422, "invalid_page", "Tamanho ou orientação de página inválidos")
    _, width, height = PAGE_SIZES[size]
    pixels = round(width / 25.4 * dpi), round(height / 25.4 * dpi)
    return pixels if orientation == "portrait" else (pixels[1], pixels[0])


def validate_layout(layout: object, page_size: tuple[int, int]) -> dict:
    if not isinstance(layout, dict) or not isinstance(layout.get("elements"), list) or not isinstance(layout.get("page"), dict):
        raise ServiceError(422, "invalid_layout", "O layout precisa conter a página e uma lista de elementos")
    page = layout["page"]
    if page_dimensions(str(page.get("size")), str(page.get("orientation"))) != tuple(page_size):
        raise ServiceError(422, "invalid_layout", "O layout não corresponde ao tamanho da página")
    if len(layout["elements"]) > 400:
        raise ServiceError(422, "invalid_layout", "O layout aceita até 400 elementos")
    elements, identifiers = [], set()
    for raw in layout["elements"]:
        if not isinstance(raw, dict) or raw.get("kind") not in LAYOUT_KINDS:
            raise ServiceError(422, "invalid_layout", "Elemento de layout inválido")
        identifier = str(raw.get("id", "")).strip()
        if not identifier or len(identifier) > 100 or identifier in identifiers:
            raise ServiceError(422, "invalid_layout", "IDs de elementos de layout devem ser únicos")
        identifiers.add(identifier)
        box = raw.get("bbox")
        if not isinstance(box, list) or len(box) != 4 or any(type(item) is not int for item in box):
            raise ServiceError(422, "invalid_layout", "Toda área deve ter quatro coordenadas inteiras")
        x1, y1, x2, y2 = box
        if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1 or x2 > page_size[0] or y2 > page_size[1]:
            raise ServiceError(422, "invalid_layout", "Um elemento de layout está fora da página")
        element = {"id": identifier, "kind": raw["kind"], "bbox": box}
        if raw["kind"] == "text":
            text, size = raw.get("text", ""), raw.get("font_size", 10)
            if not isinstance(text, str) or len(text) > 1000:
                raise ServiceError(422, "invalid_layout", "O texto fixo aceita até 1000 caracteres")
            if isinstance(size, bool) or not isinstance(size, (int, float)) or not 4 <= size <= 72:
                raise ServiceError(422, "invalid_layout", "O tamanho da fonte precisa estar entre 4 e 72 pt")
            if raw.get("align", "left") not in LAYOUT_ALIGNMENTS:
                raise ServiceError(422, "invalid_layout", "Alinhamento de texto inválido")
            element.update(text=text, font_size=size, bold=raw.get("bold") is True, align=raw.get("align", "left"))
        else:
            stroke = raw.get("stroke", 1)
            if isinstance(stroke, bool) or not isinstance(stroke, (int, float)) or not 0.25 <= stroke <= 10:
                raise ServiceError(422, "invalid_layout", "A espessura precisa estar entre 0,25 e 10 pt")
            element["stroke"] = stroke
            if raw["kind"] == "box":
                element["fill"] = raw.get("fill") is True
        elements.append(element)
    return {"page": {"size": page["size"], "orientation": page["orientation"]}, "elements": elements}


def render_layout_pdf(layout: dict, page_size: tuple[int, int], dpi: int = 300) -> bytes:
    import fitz

    scale = 72 / dpi
    with fitz.open() as document:
        page = document.new_page(width=page_size[0] * scale, height=page_size[1] * scale)
        for element in layout["elements"]:
            x1, y1, x2, y2 = (value * scale for value in element["bbox"])
            if element["kind"] == "text":
                size = float(element["font_size"])
                font = "hebo" if element.get("bold") else "helv"
                for index, line in enumerate(element["text"].split("\n")):
                    width = fitz.get_text_length(line, fontname=font, fontsize=size)
                    x = {"left": x1, "center": (x1 + x2 - width) / 2, "right": x2 - width}[element["align"]]
                    page.insert_text((x, y1 + size * 0.92 + index * size * 1.2), line, fontname=font, fontsize=size)
            elif element["kind"] == "line":
                if x2 - x1 >= y2 - y1:
                    page.draw_line((x1, (y1 + y2) / 2), (x2, (y1 + y2) / 2), width=element["stroke"])
                else:
                    page.draw_line(((x1 + x2) / 2, y1), ((x1 + x2) / 2, y2), width=element["stroke"])
            else:
                page.draw_rect(fitz.Rect(x1, y1, x2, y2), width=element["stroke"],
                               fill=(0.93, 0.94, 0.95) if element.get("fill") else None)
        return document.tobytes(garbage=3, deflate=True)


def revision(*paths: Path) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.read_bytes() if path.is_file() else b"")
    return digest.hexdigest()[:16]


def semantic_errors(field: dict, document: str | None = None) -> list[str]:
    if field.get("semantic") in (None, ""):
        return []
    decision = semantic_plan([field], document).decisions[str(field.get("field_id", ""))]
    return [decision.issue] if decision.invalid else []


def semantic_inferred(definition: dict, document: str | None) -> dict[str, str | None]:
    found = {}
    for field in apply_shared_rules(definition)["fields"]:
        decision = semantic_plan([{**field, "semantic": None}], document).decisions[str(field.get("field_id", ""))]
        found[str(field.get("field_id", ""))] = decision.binding.key if decision.binding else None
    return found


def mapping_issues(definition: dict) -> list[str]:
    fields = definition.get("fields") or []
    issues = [] if fields else ["Adicione ao menos um campo ao documento"]
    for field in fields:
        name, data_type = field.get("label") or field["field_id"], field.get("data_type", "text")
        validation = field.get("validation") or {}
        if data_type == "checkbox" and len(field.get("elements") or []) < 2:
            issues.append(f"{name}: o checkbox precisa de ao menos duas opções desenhadas")
        if data_type in COMPOSITE_TYPES - {"checkbox"} and not field.get("elements") and not validation.get("digit_count"):
            issues.append(f"{name}: a sequência precisa de caixas de caracteres")
        if data_type == "enum" and not validation.get("allowed_values"):
            issues.append(f"{name}: informe os valores permitidos da lista")
        if data_type == "date" and validation.get("format") == SEPARATED_DATE:
            parts = sorted(str(element.get("part")) for element in field.get("elements") or [])
            if parts != sorted(DATE_PARTS):
                issues.append(f"{name}: a data separada precisa de uma área para o dia, uma para o mês e uma para o ano")
        if data_type == "dnv_number" and validation.get("format") == "caixas" and len(field.get("elements") or []) != 11:
            issues.append(f"{name}: o número da DNV em caixas precisa de 11 caixas, uma por algarismo")
    if fields:
        issues.extend(semantic_plan(apply_shared_rules(definition)["fields"], definition.get("template_id")).issues)
    return issues


def preview_generation(template: Path, definition: dict, seed: int) -> str:
    import fitz
    from src.synthetic_dataset import generate_dataset

    if not definition["fields"]:
        raise ServiceError(422, "empty_mapping", "Adicione ao menos um campo para pré-visualizar")
    with tempfile.TemporaryDirectory(prefix="ocrsd-preview-") as directory:
        mapping = Path(directory) / "mapping.json"
        mapping.write_text(json.dumps(definition, ensure_ascii=False), encoding="utf-8")
        try:
            documents = generate_dataset(template, mapping, {field["field_id"] for field in definition["fields"]},
                                         Path(directory) / "saida", 1, seed)
        except ValueError as exc:
            raise ServiceError(422, "preview_failed", str(exc)) from exc
        mark_pdf(documents[0].pdf_path)
        with fitz.open(documents[0].pdf_path) as document:
            page = document[0]
            scale = min(1400 / page.rect.width, 2.2)
            image = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False).tobytes("png")
    return "data:image/png;base64," + base64.b64encode(image).decode()


def template_page_size(pdf: bytes, dpi: int = 300) -> tuple[int, int]:
    import fitz

    try:
        with fitz.open(stream=pdf, filetype="pdf") as document:
            if document.page_count != 1:
                raise ServiceError(422, "invalid_pdf", "Use um PDF de página única")
            page = document[0]
            scale = dpi / 72
            return round(page.rect.width * scale), round(page.rect.height * scale)
    except ServiceError:
        raise
    except Exception:
        raise ServiceError(422, "invalid_pdf", "O arquivo não é um PDF válido")


def render_preview(template: Path, max_width: int = 1200) -> bytes:
    import fitz

    with fitz.open(template) as document:
        page = document[0]
        scale = min(max_width / page.rect.width, 2.2)
        return page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False).tobytes("png")


def pdf_page_count(path: Path) -> int:
    import fitz

    with fitz.open(path) as document:
        return len(document)


def render_pdf_page(path: Path, page_number: int, width: int = 1800) -> bytes:
    import fitz

    with fitz.open(path) as document:
        if page_number < 1 or page_number > len(document):
            raise ServiceError(404, "not_found", "Página não encontrada")
        page = document[page_number - 1]
        scale = width / page.rect.width
        return page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False).tobytes("png")


PREVIEW_SAMPLES = (("Nome", "Ana Carolina Silva"), ("Data", "15/09/2026"), ("CPF", "123.456.789-09"),
                   ("Texto alfanumérico", "13H00 M00 S"))
PREVIEW_RENDERER = "ocrsd-preview-1"
_MODEL_LOCK = threading.Lock()


@functools.lru_cache(maxsize=2)
def cached_handwriting_model(name: str):
    from src.neural_handwriting import load_model

    return load_model(name)


def render_writing_samples(writing: Writing, texts: list[str], variation: int = 0) -> list[dict]:
    from PIL import Image
    from src.neural_handwriting import NeuralPen, prepare_text
    from src.synthetic_dataset import (is_numeric_text, random_handwriting_style, render_handwritten_numeric,
                                       render_handwritten_text, render_neural_text)

    variation_rng = random.Random(42 + variation * 7919)
    writer = writing.writer if writing.writer is not None else variation_rng.randrange(13 if writing.model == "sjvasquez" else 512)
    results = []
    for index, text in enumerate(texts):
        seed = 10_000 + variation * 1000 + index * 100 + len(text)
        result = {"text": text, "rendered": text, "image": None, "error": None, "writer": None if writing.model == "font" else writer}
        try:
            if writing.model == "font":
                numeric = is_numeric_text(text)
                rng = random.Random(seed)
                style = random_handwriting_style(rng, intensity=writing.intensity, medium="pen" if numeric else "pencil")
                image = (render_handwritten_numeric(text, 520, 116, rng, style=style, intensity=writing.intensity) if numeric
                         else render_handwritten_text(text, 520, 116, rng, style=style, intensity=writing.intensity, medium="pencil"))
            else:
                rendered = prepare_text(text, writing.text_policy)
                pen = NeuralPen(model=writing.model, writer=writer, bias=writing.bias, color=writing.color, width=writing.width,
                                pressure=writing.pressure, dropout=writing.dropout, text_policy=writing.text_policy)
                with _MODEL_LOCK:
                    image, _ = render_neural_text(rendered, 520, 116, pen, cached_handwriting_model(writing.model), random.Random(seed))
                result["rendered"] = rendered
        except (ValueError, FileNotFoundError) as exc:
            result["error"] = ("O modelo não escreve todos os caracteres deste texto com a política “Rejeitar”"
                               if "cannot write" in str(exc) else "Não foi possível renderizar esta amostra")
            results.append(result)
            continue
        white = Image.new("RGBA", image.size, "white")
        white.alpha_composite(image.convert("RGBA"))
        buffer = io.BytesIO()
        white.convert("RGB").save(buffer, format="PNG")
        result["image"] = buffer.getvalue()
        results.append(result)
    return results


def handwriting_previews(writing: Writing, texts: list[str]) -> list[str]:
    images = []
    for sample in render_writing_samples(writing, texts):
        if sample["image"] is None:
            raise ValueError(sample["error"])
        images.append("data:image/png;base64," + base64.b64encode(sample["image"]).decode())
    return images


def preview_key(writing: Writing, rows: int) -> str:
    payload = json.dumps({"writing": writing.model_dump(), "rows": rows, "samples": PREVIEW_SAMPLES, "renderer": PREVIEW_RENDERER}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:24]


def preview_strip(writing: Writing, cache: Path, rows: int = 4) -> bytes:
    from PIL import Image, ImageDraw

    path = cache / f"{preview_key(writing, rows)}.png"
    if path.is_file():
        return path.read_bytes()
    samples = render_writing_samples(writing, [text for _, text in PREVIEW_SAMPLES[:rows]])
    sheet = Image.new("RGB", (520, 116 * rows), "white")
    draw = ImageDraw.Draw(sheet)
    for index, sample in enumerate(samples):
        if sample["image"]:
            sheet.paste(Image.open(io.BytesIO(sample["image"])), (0, index * 116))
        else:
            draw.rounded_rectangle((16, index * 116 + 24, 504, index * 116 + 92), radius=10, outline=(203, 213, 225), width=2)
            draw.line((40, index * 116 + 58, 480, index * 116 + 58), fill=(203, 213, 225), width=2)
    buffer = io.BytesIO()
    sheet.save(buffer, format="PNG", optimize=True)
    cache.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_bytes(buffer.getvalue())
    temporary.replace(path)
    return buffer.getvalue()


def realism_previews(template: Path, preset: str, seed: int = 42) -> tuple[str, str, dict]:
    import cv2
    import fitz
    import numpy as np
    from src.document_filters import apply_document_filters

    with fitz.open(template) as document:
        pixmap = document[0].get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
    original = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(pixmap.height, pixmap.width, pixmap.n)
    original = cv2.cvtColor(original[:, :, :3], cv2.COLOR_RGB2BGR)
    filtered, metadata = apply_document_filters(original, random.Random(seed), preset)
    def encoded(image) -> str:
        ok, content = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 88])
        if not ok:
            raise ServiceError(500, "preview_failed", "Não foi possível produzir a prévia")
        return "data:image/jpeg;base64," + base64.b64encode(content).decode()
    return encoded(original), encoded(filtered), metadata


def generate(options: GenerationOptions, output: Path, settings: Settings,
             progress: Callable[[int], None], owner_id: str | None = None,
             document_ready: Callable[[int, Path, Path], None] | None = None, source: dict | None = None) -> list[dict]:
    with tempfile.TemporaryDirectory(prefix="ocrsd-source-") as directory:
        catalog = Catalog(settings)
        if source:
            options = catalog.validate(options, owner_id, source["definition"])
            template, mapping = source["template"], Path(directory) / f"{options.mapping_id}.json"
            mapping.write_text(json.dumps(source["definition"], ensure_ascii=False), encoding="utf-8")
        else:
            options = catalog.validate(options, owner_id)
            template = catalog.document(options.document_type, owner_id).template
            mapping = catalog.mapping(options.document_type, options.mapping_id, owner_id)
        return generate_from(options, template, mapping, output, progress, document_ready)


def generate_from(options: GenerationOptions, template: Path, mapping: Path, output: Path, progress: Callable[[int], None],
                  document_ready: Callable[[int, Path, Path], None] | None = None) -> list[dict]:
    from src.neural_handwriting import NeuralPen
    from src.synthetic_dataset import generate_dataset

    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    writing = options.writing
    pen = None if writing.model == "font" or not (writing.handwriting or writing.text_handwriting) else NeuralPen(**writing.model_dump(exclude={"handwriting", "text_handwriting", "intensity"}))
    def finalize_document(index: int, total: int, document) -> None:
        mark_pdf(document.pdf_path)
        data = json.loads(document.json_path.read_text())
        data.update(source_template=options.document_type, synthetic=True, notice=NOTICE, generation_seed=options.seed)
        temporary = document.json_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(document.json_path)
        if document_ready:
            document_ready(index, document.pdf_path, document.json_path)
        progress(index)

    documents = generate_dataset(
        template, mapping, set(options.fields), output, options.count, options.seed,
        handwriting=writing.handwriting, text_handwriting=writing.text_handwriting,
        handwriting_intensity=writing.intensity, neural_pen=pen,
        realistic_filters=options.filter_preset != "none", filter_preset=options.filter_preset if options.filter_preset != "none" else "medium",
        progress_callback=finalize_document,
    )
    manifest_path = output / "dataset.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.update(template=options.document_type, mapping=options.mapping_id, seed=options.seed, synthetic=True, notice=NOTICE,
                    options=options.model_dump())
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    files = sorted(path for path in output.iterdir() if path.suffix in {".pdf", ".json"})
    with ZipFile(output / "dataset.zip", "w", ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.name)
    files.append(output / "dataset.zip")
    types = {".pdf": "application/pdf", ".json": "application/json", ".zip": "application/zip"}
    return [{"id": path.name, "name": path.name,
             "media_type": types[path.suffix], "size": path.stat().st_size}
            for path in files]
