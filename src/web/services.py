from __future__ import annotations

import json
import base64
import io
from pathlib import Path
import random
import sqlite3
import tempfile
from typing import Callable
from zipfile import ZIP_DEFLATED, ZipFile

from src.document_types import discover_document_types
from src.web.models import DocumentModelResponse, GenerationOptions, PresetResponse, Writing
from src.web.settings import Settings
from src.writing_presets import list_presets

NOTICE = "DOCUMENTO SINTÉTICO — SEM VALIDADE LEGAL"


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

    def save_builtin_mapping(self, key: str, mapping_id: str, definition: dict) -> None:
        if not self.settings.catalog_writable:
            raise ServiceError(403, "catalog_read_only", "O catálogo instalado está em modo somente leitura")
        path = self.mapping(key, mapping_id)
        current = json.loads(path.read_text(encoding="utf-8"))
        definition = {**definition, "source_template": current.get("source_template", str(self.document(key).template))}
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False) as temporary:
            temporary.write(json.dumps(definition, ensure_ascii=False, indent=2) + "\n")
            temporary_path = Path(temporary.name)
        temporary_path.replace(path)

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
        result = []
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
        result = [{"id": item.key, "label": item.label,
                   "mappings": [{"id": path.stem, "label": path.stem.replace("_", " ")} for path in item.mappings]}
                  for item in self.types.values()]
        result.extend({"id": row["id"], "label": row["label"],
                       "mappings": [{"id": row["mapping_id"], "label": row["mapping_id"].replace("_", " ")}]}
                      for row in custom)
        return result

    def document_models(self, custom: list[dict]) -> list[DocumentModelResponse]:
        builtins = []
        for item in self.types.values():
            definition = json.loads(item.mappings[0].read_text(encoding="utf-8"))
            builtins.append(DocumentModelResponse(id=item.key, label=item.label, description="Modelo incluído no catálogo OCRSD",
                                                  status="published", version=1, field_count=len(definition["fields"]),
                                                  mapping_id=item.mappings[0].stem, builtin=True, editable=self.settings.catalog_writable))
        return builtins + [custom_model(row) for row in custom]

    def validate(self, options: GenerationOptions, owner_id: str | None = None) -> GenerationOptions:
        definition = self.definition(options.document_type, options.mapping_id, owner_id)
        fields = {str(field["field_id"]) for field in definition["fields"]}
        if set(options.fields) - fields or len(set(options.fields)) != len(options.fields):
            raise ServiceError(422, "invalid_fields", "Selecione campos existentes, sem repetições")
        if options.count > self.settings.max_count:
            raise ServiceError(422, "quota", f"Limite de {self.settings.max_count} documentos por job")
        if not any(m["id"] == options.writing.model and m["available"] for m in self.models()):
            raise ServiceError(422, "model_unavailable", "Modelo não habilitado ou pesos ausentes")
        return options


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


def duplicate_model(catalog: Catalog, store, model_id: str, owner_id: str) -> dict:
    if model_id in catalog.types:
        document = catalog.document(model_id)
        mapping_id = document.mappings[0].stem
        definition = catalog.definition(model_id, mapping_id)
        label = document.label + " — cópia"
    else:
        source = store.model(model_id, owner_id)
        document = catalog.document(model_id, owner_id)
        mapping_id = source["mapping_id"]
        definition = catalog.definition(model_id, mapping_id, owner_id)
        label = source["label"] + " — cópia"
    definition["source_template"] = "template.pdf"
    return store.create_model(owner_id, label, "Cópia editável", mapping_id,
                              document.template.read_bytes(), validate_mapping(definition))


def save_mapping(catalog: Catalog, store, model_id: str, owner_id: str, definition: dict) -> str:
    if model_id in catalog.types:
        mapping_id = catalog.document(model_id).mappings[0].stem
        current = catalog.definition(model_id, mapping_id)
        catalog.save_builtin_mapping(model_id, mapping_id, validate_mapping(definition, tuple(current["page_size_px"])))
        return mapping_id
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
        field = {**raw, "field_id": identifier, "label": str(raw.get("label") or identifier), "bbox": bbox(raw.get("bbox"))}
        elements = []
        element_ids: set[str] = set()
        for raw_element in raw.get("elements", []):
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


def handwriting_previews(writing: Writing, texts: list[str]) -> list[str]:
    from PIL import Image
    from src.neural_handwriting import NeuralPen, load_model, prepare_text
    from src.synthetic_dataset import random_handwriting_style, render_handwritten_numeric, render_handwritten_text, render_neural_text

    model = None if writing.model == "font" else load_model(writing.model)
    writer = writing.writer if writing.writer is not None else random.Random(42).randrange(13 if writing.model == "sjvasquez" else 512)
    images = []
    for index, text in enumerate(texts):
        seed = 10_000 + index * 100 + len(text)
        if writing.model == "font":
            style = random_handwriting_style(random.Random(seed), intensity=writing.intensity, medium="pen")
            image = (render_handwritten_numeric(text, 520, 116, random.Random(seed), style=style, intensity=writing.intensity)
                     if all(character in "0123456789.,:/- " for character in text)
                     else render_handwritten_text(text, 520, 116, random.Random(seed), style=style, intensity=writing.intensity, medium="pen"))
            alpha = image.getchannel("A") if image.mode == "RGBA" else image.convert("L")
            color = tuple(int(writing.color[offset:offset + 2], 16) for offset in (1, 3, 5))
            colored = Image.new("RGBA", image.size, (*color, 0))
            colored.putalpha(alpha)
            image = colored
        else:
            pen = NeuralPen(model=writing.model, writer=writer, bias=writing.bias, color=writing.color, width=writing.width,
                            pressure=writing.pressure, dropout=writing.dropout, text_policy=writing.text_policy)
            image, _ = render_neural_text(prepare_text(text, writing.text_policy), 520, 116, pen, model, random.Random(seed))
        white = Image.new("RGBA", image.size, "white")
        white.alpha_composite(image.convert("RGBA"))
        buffer = io.BytesIO()
        white.convert("RGB").save(buffer, format="PNG")
        images.append("data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode())
    return images


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
             progress: Callable[[int], None], owner_id: str | None = None) -> list[dict]:
    from src.neural_handwriting import NeuralPen
    from src.synthetic_dataset import generate_dataset

    catalog = Catalog(settings)
    catalog.validate(options, owner_id)
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    writing = options.writing
    pen = None if writing.model == "font" or not (writing.handwriting or writing.text_handwriting) else NeuralPen(**writing.model_dump(exclude={"handwriting", "text_handwriting", "intensity"}))
    documents = generate_dataset(
        catalog.document(options.document_type, owner_id).template, catalog.mapping(options.document_type, options.mapping_id, owner_id),
        set(options.fields), output, options.count, options.seed,
        handwriting=writing.handwriting, text_handwriting=writing.text_handwriting,
        handwriting_intensity=writing.intensity, neural_pen=pen,
        realistic_filters=options.filter_preset != "none", filter_preset=options.filter_preset if options.filter_preset != "none" else "medium",
        progress_callback=lambda index, total, document: progress(index),
    )
    for document in documents:
        mark_pdf(document.pdf_path)
        data = json.loads(document.json_path.read_text())
        data["source_template"] = options.document_type
        data["synthetic"] = True
        data["notice"] = NOTICE
        document.json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest_path = output / "dataset.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.update(template=options.document_type, mapping=options.mapping_id, synthetic=True, notice=NOTICE, options=options.model_dump())
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    files = sorted(output.iterdir())
    with ZipFile(output / "dataset.zip", "w", ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.name)
    files.append(output / "dataset.zip")
    types = {".pdf": "application/pdf", ".json": "application/json", ".zip": "application/zip"}
    return [{"id": str(index), "name": path.name, "media_type": types[path.suffix], "size": path.stat().st_size}
            for index, path in enumerate(files)]
