from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import re


DOCUMENTS_ROOT = Path(__file__).resolve().parent.parent / "documents"


@dataclass(frozen=True)
class DocumentType:
    key: str
    root: Path
    template: Path
    mappings: tuple[Path, ...]
    presets: tuple[Path, ...]
    dataset: Path

    @property
    def metadata(self) -> dict:
        path = self.root / "modelo.json"
        try:
            data = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        except (OSError, ValueError):
            data = {}
        return data if isinstance(data, dict) else {}

    @property
    def label(self) -> str:
        label = self.metadata.get("label")
        return label.strip() if isinstance(label, str) and label.strip() else self.key.replace("_", " ").upper()

    @property
    def description(self) -> str:
        description = self.metadata.get("description")
        return description.strip() if isinstance(description, str) else ""

    @property
    def active_mapping(self) -> Path:
        name = self.metadata.get("active_mapping")
        return next((path for path in self.mappings if path.stem == name), self.mappings[0])


def discover_document_types(root: Path = DOCUMENTS_ROOT) -> dict[str, DocumentType]:
    if not root.is_dir():
        return {}
    discovered: dict[str, DocumentType] = {}
    for folder in sorted(path for path in root.iterdir() if path.is_dir()):
        template = _template_for_folder(folder)
        if not template.is_file():
            continue
        mappings_dir = folder / "mappings"
        presets_dir = folder / "presets"
        mappings = tuple(sorted(mappings_dir.glob("*.json")))
        if not mappings:
            mappings = tuple(sorted(folder.glob("*_fields.json")))
        if not mappings:
            continue
        discovered[folder.name] = DocumentType(
            key=folder.name,
            root=folder,
            template=template,
            mappings=mappings,
            presets=tuple(sorted(presets_dir.glob("*.json"))),
            dataset=folder / "dataset",
        )
    return discovered


def _template_for_folder(folder: Path) -> Path:
    template = folder / "template.pdf"
    if template.is_file():
        return template
    named_template = folder / f"{folder.name}.pdf"
    if named_template.is_file():
        return named_template
    pdfs = sorted(folder.glob("*.pdf"))
    return pdfs[0] if pdfs else template


def safe_document_type_key(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", value).strip("_").lower()
    return cleaned or "documento"


def infer_document_type_key(template_path: Path | None = None, mapping_path: Path | None = None,
                            root: Path = DOCUMENTS_ROOT) -> str:
    for path in (template_path, mapping_path):
        if path is None:
            continue
        resolved = path.resolve()
        for parent in (resolved.parent, *resolved.parents):
            if parent.parent == root.resolve():
                return safe_document_type_key(parent.name)
    if template_path is not None and template_path.parent.name:
        return safe_document_type_key(template_path.parent.name)
    return "documento"


def build_dataset_run_dir(document_type_key: str, count: int, *,
                          root: Path = DOCUMENTS_ROOT,
                          now: datetime | None = None) -> Path:
    timestamp = (now or datetime.now()).strftime("%Y%m%d_%H%M")
    key = safe_document_type_key(document_type_key)
    base = root / key / "dataset" / f"{timestamp}-{key}-{count}"
    if not base.exists():
        return base
    for suffix in range(2, 1000):
        candidate = base.with_name(f"{base.name}_{suffix:02d}")
        if not candidate.exists():
            return candidate
    raise FileExistsError(f"Não foi possível criar uma pasta única para {base}")


def collect_dataset_pdfs(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(
        path for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() == ".pdf"
    )
