from __future__ import annotations

import json
from pathlib import Path
from typing import Callable
from zipfile import ZIP_DEFLATED, ZipFile

from src.document_types import discover_document_types
from src.web.models import GenerationOptions, PresetResponse, Writing
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

    def document(self, key: str):
        if key not in self.types:
            raise ServiceError(404, "not_found", "Tipo de documento não encontrado")
        return self.types[key]

    def mapping(self, key: str, mapping_id: str) -> Path:
        for path in self.document(key).mappings:
            if path.stem == mapping_id:
                return path
        raise ServiceError(404, "not_found", "Mapeamento não encontrado")

    def definition(self, key: str, mapping_id: str) -> dict:
        definition = json.loads(self.mapping(key, mapping_id).read_text(encoding="utf-8"))
        definition["source_template"] = key
        return definition

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

    def validate(self, options: GenerationOptions) -> GenerationOptions:
        definition = self.definition(options.document_type, options.mapping_id)
        fields = {str(field["field_id"]) for field in definition["fields"]}
        if set(options.fields) - fields or len(set(options.fields)) != len(options.fields):
            raise ServiceError(422, "invalid_fields", "Selecione campos existentes, sem repetições")
        if options.count > self.settings.max_count:
            raise ServiceError(422, "quota", f"Limite de {self.settings.max_count} documentos por job")
        if not any(m["id"] == options.writing.model and m["available"] for m in self.models()):
            raise ServiceError(422, "model_unavailable", "Modelo não habilitado ou pesos ausentes")
        return options


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
        document.saveIncr()


def generate(options: GenerationOptions, output: Path, settings: Settings,
             progress: Callable[[int], None]) -> list[dict]:
    from src.neural_handwriting import NeuralPen
    from src.synthetic_dataset import generate_dataset

    catalog = Catalog(settings)
    catalog.validate(options)
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    writing = options.writing
    pen = None if writing.model == "font" or not (writing.handwriting or writing.text_handwriting) else NeuralPen(**writing.model_dump(exclude={"handwriting", "text_handwriting", "intensity"}))
    documents = generate_dataset(
        catalog.document(options.document_type).template, catalog.mapping(options.document_type, options.mapping_id),
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
