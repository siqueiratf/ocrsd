from __future__ import annotations

import json
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PRESET_LIBRARY_PATH = PROJECT_ROOT / "documents" / "writing_presets.json"
UID_SEPARATOR = "__style-"
UID_RE = re.compile(r"__style-([A-Za-z0-9_-]+)$")


def _default_library() -> dict[str, Any]:
    return {"version": 1, "presets": []}


def load_preset_library(path: Path | None = None) -> dict[str, Any]:
    path = path or PRESET_LIBRARY_PATH
    if not path.is_file():
        return _default_library()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return _default_library()
    presets = data.get("presets")
    if not isinstance(presets, list):
        return _default_library()
    return {"version": int(data.get("version", 1)), "presets": presets}


def save_preset_library(library: dict[str, Any], path: Path | None = None) -> None:
    path = path or PRESET_LIBRARY_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(library, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def new_preset_uid(prefix: str = "preset") -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{stamp}_{uuid.uuid4().hex[:6]}"


def list_presets(path: Path | None = None) -> list[dict[str, Any]]:
    return [preset for preset in load_preset_library(path).get("presets", []) if isinstance(preset, dict)]


def find_preset(uid: str | None, path: Path | None = None) -> dict[str, Any] | None:
    if not uid:
        return None
    for preset in list_presets(path):
        if preset.get("uid") == uid:
            return preset
    return None


def upsert_preset(preset: dict[str, Any], path: Path | None = None) -> dict[str, Any]:
    if not preset.get("uid"):
        preset = {**preset, "uid": new_preset_uid()}
    library = load_preset_library(path)
    presets = [item for item in library.get("presets", []) if item.get("uid") != preset["uid"]]
    presets.append(preset)
    library["presets"] = presets
    save_preset_library(library, path)
    return preset


def uid_from_pdf_name(name: str) -> str | None:
    stem = Path(name).stem
    match = UID_RE.search(stem)
    return match.group(1) if match else None


def document_id_with_preset(document_id: str, preset_uid: str | None) -> str:
    if not preset_uid:
        return document_id
    safe_uid = re.sub(r"[^A-Za-z0-9_-]+", "_", preset_uid).strip("_")
    return f"{document_id}{UID_SEPARATOR}{safe_uid}" if safe_uid else document_id


def generation_settings(
    *,
    model: str,
    writer: str | int | None,
    bias: float,
    pen_color: str,
    pen_width: float,
    pen_pressure: float,
    pen_dropout: float,
    text_policy: str,
    handwriting: bool = True,
    text_handwriting: bool = True,
    handwriting_intensity: float = 0.65,
    realistic_filters: bool = False,
    filter_preset: str = "medium",
) -> dict[str, Any]:
    return {
        "model": model,
        "writer": "Aleatório" if writer in (None, "", "Aleatório") else str(writer),
        "bias": float(bias),
        "pen_color": pen_color,
        "pen_width": float(pen_width),
        "pen_pressure": float(pen_pressure),
        "pen_dropout": float(pen_dropout),
        "text_policy": text_policy,
        "handwriting": bool(handwriting),
        "text_handwriting": bool(text_handwriting),
        "handwriting_intensity": float(handwriting_intensity),
        "realistic_filters": bool(realistic_filters),
        "filter_preset": filter_preset,
    }


def _name_part(value: object) -> str:
    text = str(value).strip().lower()
    text = text.replace("#", "")
    text = text.replace("aleatório", "aleatorio")
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-") or "na"


def _compact_number(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return _name_part(value)
    text = f"{number:.2f}".rstrip("0").rstrip(".")
    return text.replace(".", "p")


def suggested_preset_name(settings: dict[str, Any]) -> str:
    if settings.get("mode") == "mixed":
        numbers = settings.get("numbers") or {}
        text = settings.get("text") or {}
        return "_".join([
            "mixed",
            "num", _single_preset_name(numbers),
            "txt", _single_preset_name(text),
            f"filtro-{_name_part(settings.get('filter_preset', 'medium'))}"
            if settings.get("realistic_filters") else "sem-filtro",
        ])
    return _single_preset_name(settings)


def _single_preset_name(settings: dict[str, Any]) -> str:
    model = _name_part(settings.get("model", "font"))
    writer = _name_part(settings.get("writer", "aleatorio"))
    color = _name_part(settings.get("pen_color", ""))
    pieces = [
        model,
        f"estilo-{writer}",
        f"reg-{_compact_number(settings.get('bias', 0))}",
        f"cor-{color}" if color else "cor-padrao",
        f"esp-{_compact_number(settings.get('pen_width', 0))}",
        f"press-{_compact_number(settings.get('pen_pressure', 0))}",
        f"falha-{_compact_number(settings.get('pen_dropout', 0))}",
        _name_part(settings.get("text_policy", "strict")),
    ]
    if settings.get("realistic_filters"):
        pieces.append(f"filtro-{_name_part(settings.get('filter_preset', 'medium'))}")
    return "_".join(pieces)


def preset_summary(preset: dict[str, Any] | None) -> str:
    if not preset:
        return "Preset de escrita não encontrado."
    settings = preset.get("settings") or {}
    if settings.get("mode") == "mixed":
        numbers = settings.get("numbers") or {}
        text = settings.get("text") or {}
        return "\n".join([
            f"Preset: {preset.get('name') or preset.get('uid')}",
            f"UID: {preset.get('uid', '')}",
            f"Números: {numbers.get('model', '')} · estilo {numbers.get('writer', 'Aleatório')}",
            f"Caneta números: {numbers.get('pen_color', '')}, {numbers.get('pen_width', '')} px",
            f"Textos: {text.get('model', '')} · estilo {text.get('writer', 'Aleatório')}",
            f"Caneta textos: {text.get('pen_color', '')}, {text.get('pen_width', '')} px",
            f"Regularidade: números={numbers.get('bias', '')}, textos={text.get('bias', '')}",
            f"Filtros: {settings.get('realistic_filters', '')} ({settings.get('filter_preset', '')})",
        ])
    lines = [
        f"Preset: {preset.get('name') or preset.get('uid')}",
        f"UID: {preset.get('uid', '')}",
        f"Modelo: {settings.get('model', '')}",
        f"Estilo: {settings.get('writer', 'Aleatório')}",
        f"Regularidade: {settings.get('bias', '')}",
        f"Caneta: {settings.get('pen_color', '')}, {settings.get('pen_width', '')} px",
        f"Pressão/falhas: {settings.get('pen_pressure', '')} / {settings.get('pen_dropout', '')}",
        f"Texto: {settings.get('text_policy', '')}",
        f"Manuscrito: números={settings.get('handwriting', '')}, textos={settings.get('text_handwriting', '')}",
        f"Filtros: {settings.get('realistic_filters', '')} ({settings.get('filter_preset', '')})",
    ]
    return "\n".join(lines)
