#!/usr/bin/env python3
"""Writing/rendering audit tool for OCRSD (step 0 of the quality workflow).

Renders every document type with every writing preset (fixed seeds), measures
fill/data quality and visual placement/legibility per field, and writes
comparable evidence to var/audit/<label>/. Does NOT change generator
behaviour: it only calls src.synthetic_dataset.generate_document with the
mapping each document already ships, the way src/web/services.py builds its
options (see generate_from in services.py), and inspects the output.

Run (from repo root):
    uv run --offline --no-sync python scripts/writing_audit.py --label baseline

See report.md in the output directory for findings.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import random
import re
import statistics
import subprocess
import sys
import time
import unicodedata
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import fitz  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from src.content_classes import classify_content  # noqa: E402
from src.document_types import discover_document_types  # noqa: E402
from src.field_rules import apply_shared_rules  # noqa: E402
from src.synthetic_dataset import generate_document  # noqa: E402
from src.writing_presets import list_presets  # noqa: E402
from src.semantic.assisted import ai_coverage  # noqa: E402

try:
    from src.neural_handwriting import NeuralPen
except Exception:  # pragma: no cover
    NeuralPen = None  # type: ignore

try:
    import easyocr as _easyocr
except Exception:  # pragma: no cover
    _easyocr = None  # type: ignore

_EASYOCR_READER = None


def easyocr_available() -> bool:
    return _easyocr is not None


def get_easyocr_reader():
    """Lazily build a single shared easyocr Reader (pt). Expensive to construct,
    so we build it once per process, not per field."""
    global _EASYOCR_READER
    if _easyocr is None:
        return None
    if _EASYOCR_READER is None:
        _EASYOCR_READER = _easyocr.Reader(["pt"], gpu=False, verbose=False)
    return _EASYOCR_READER


def run_easyocr(img: Image.Image) -> str:
    reader = get_easyocr_reader()
    if reader is None:
        return ""
    try:
        arr = np.array(img.convert("RGB"))
        result = reader.readtext(arr, detail=0, paragraph=False)
        return " ".join(result).strip()
    except Exception:
        return ""

DOCUMENTS_ROOT = ROOT / "documents"
ASSETS = ROOT / "assets" / "handwriting"
TESSERACT_BIN = "/usr/bin/tesseract"

STANDARD_PRESETS = [
    ("padrao_fonte_natural", "font", "caveat", 0.65, "none"),
    ("padrao_fonte_regular", "font", "caveat", 0.25, "none"),
    ("padrao_fonte_variada", "font", "caveat", 0.95, "none"),
    ("padrao_fonte_homemade_apple", "font", "homemade_apple", 0.65, "none"),
    ("padrao_digitada_courier_prime", "font", "courier_prime", 0.65, "none"),
    ("padrao_digitada_arimo", "font", "arimo", 0.65, "none"),
    ("padrao_digitada_tinos", "font", "tinos", 0.65, "none"),
]
# Digitada presets use a typed font (no real handwriting synthesis); we mark
# them so insert_value takes the typed-font path (see synthetic_dataset.py is_typed()).

DEFAULT_SEEDS = [1, 2]
NEURAL_SEEDS = [1]  # neural synthesis is much slower; documented reduction (see report.md)


def neural_models_available() -> dict[str, bool]:
    sj = ASSETS / "sjvasquez" / "weights.npz"
    wb = ASSETS / "weights.bin"
    sj_ok = sj.is_file() and all(
        (ASSETS / "sjvasquez" / "styles" / f"style-{i}-{kind}.npy").is_file()
        for i in range(13) for kind in ("chars", "strokes")
    )
    return {"sjvasquez": sj_ok, "aaron1259": wb.is_file(), "wpvnteam": wb.is_file()}


# Exploration presets for the content-style evidence (topic 4). They are
# audit-only: the web app and the API never offer them.
EXPLORATION_CONTENT_PRESETS = (
    {
        "id": "exp_digits_caveat_low_intensity",
        "label": "[exp] digitos caveat baixa intensidade",
        "base": {"font_family": "caveat", "intensity": 0.25},
        "content": {
            "digits": {"font_family": "caveat", "intensity": 0.1},
            "date": {"font_family": "caveat", "intensity": 0.1},
            "cpf_cnpj": {"font_family": "caveat", "intensity": 0.1},
        },
    },
    {
        "id": "exp_digits_neural_sjvasquez_highbias",
        "label": "[exp] digitos neurais sjvasquez bias alto",
        "base": {"font_family": "caveat", "intensity": 0.25},
        "content": {
            "digits": {"model": "sjvasquez", "bias": 3.0, "dropout": 0.0, "width": 1.2},
            "date": {"model": "sjvasquez", "bias": 3.0, "dropout": 0.0, "width": 1.2},
            "cpf_cnpj": {"model": "sjvasquez", "bias": 3.0, "dropout": 0.0, "width": 1.2},
        },
    },
    {
        "id": "exp_names_neural_sjvasquez_highbias",
        "label": "[exp] nomes neurais sjvasquez bias alto, digitos tipografados",
        "base": {"font_family": "courier_prime", "intensity": 0.65},
        "content": {
            "name": {"model": "sjvasquez", "bias": 3.0, "dropout": 0.0, "width": 1.2},
            "place": {"model": "sjvasquez", "bias": 3.0, "dropout": 0.0, "width": 1.2},
            "short_text": {"model": "sjvasquez", "bias": 3.0, "dropout": 0.0, "width": 1.2},
            "long_text": {"model": "sjvasquez", "bias": 3.0, "dropout": 0.0, "width": 1.2},
        },
    },
)


def build_presets(models_available: dict[str, bool]) -> list[dict]:
    """Return the list of writing presets the web app offers, filtered to
    what is actually usable on this machine. Mirrors
    src/web/services.py:Catalog.presets()/models() without needing a full
    Settings/Catalog object (we only need the writing config, not the HTTP layer)."""
    presets = []
    for uid, model, font_family, intensity, filter_preset in STANDARD_PRESETS:
        presets.append({"uid": uid, "label": uid, "model": model, "font_family": font_family,
                         "intensity": intensity, "writer": None, "bias": 0.75, "color": "#193787",
                         "width": 1.5, "pressure": 0.25, "dropout": 0.0, "text_policy": "normalize",
                         "filter_preset": filter_preset, "kind": "standard"})
    from src.web.services import CONTENT_STYLE_PRESETS
    for spec in (*CONTENT_STYLE_PRESETS, *EXPLORATION_CONTENT_PRESETS):
        has_neural_content = any(override.get("model", "font") != "font" for override in spec["content"].values())
        presets.append({"uid": spec["id"], "label": spec["label"], "model": "font",
                        "font_family": spec["base"]["font_family"], "intensity": spec["base"]["intensity"],
                        "writer": None, "bias": 0.75, "color": "#193787", "width": 1.5, "pressure": 0.25,
                        "dropout": 0.0, "text_policy": "normalize", "filter_preset": "none",
                        "kind": "neural" if has_neural_content else "standard",
                        "content": spec["content"]})
    for preset in list_presets(DOCUMENTS_ROOT / "writing_presets.json"):
        settings = preset.get("settings", {})
        if settings.get("mode") == "mixed":
            continue  # mixed numeric/text presets are desktop-only per services.py; excluded, see report.md
        model = settings.get("model", "font")
        if model != "font" and not models_available.get(model, False):
            continue
        writer = settings.get("writer")
        writer = None if writer in (None, "", "Aleatório") else int(writer)
        presets.append({
            "uid": preset["uid"], "label": preset.get("name", preset["uid"]), "model": model,
            "font_family": settings.get("font_family", "caveat"), "intensity": settings.get("handwriting_intensity", 0.65),
            "writer": writer, "bias": settings.get("bias", 0.75), "color": settings.get("pen_color", "#193787"),
            "width": settings.get("pen_width", 1.5), "pressure": settings.get("pen_pressure", 0.25),
            "dropout": settings.get("pen_dropout", 0), "text_policy": settings.get("text_policy", "normalize"),
            "filter_preset": settings.get("filter_preset", "medium") if settings.get("realistic_filters") else "none",
            "kind": "neural",
        })
    return presets


def make_pen(preset: dict):
    if preset["model"] == "font" or NeuralPen is None:
        return None
    return NeuralPen(model=preset["model"], writer=preset["writer"], bias=preset["bias"], color=preset["color"],
                      width=preset["width"], pressure=preset["pressure"], dropout=preset["dropout"],
                      text_policy=preset["text_policy"])


# ---------------------------------------------------------------------------
# Data-quality helpers (generic, data_type/semantic driven)
# ---------------------------------------------------------------------------

CPF_RE = re.compile(r"^\d{3}\.\d{3}\.\d{3}-\d{2}$")
CEP_RE = re.compile(r"^\d{5}-?\d{3}$")
UF_SET = {"AC", "AL", "AP", "AM", "BA", "CE", "DF", "ES", "GO", "MA", "MT", "MS", "MG", "PA", "PB", "PR",
          "PE", "PI", "RJ", "RN", "RS", "RO", "RR", "SC", "SP", "SE", "TO"}


def cpf_check_digits_ok(cpf: str) -> bool | None:
    digits = re.sub(r"\D", "", cpf)
    if len(digits) != 11 or len(set(digits)) == 1:
        return None
    nums = [int(d) for d in digits]
    for size in (9, 10):
        total = sum(nums[i] * (size + 1 - i) for i in range(size))
        rem = (total * 10) % 11
        rem = 0 if rem == 10 else rem
        if rem != nums[size]:
            return False
    return True


def validate_value(data_type: str, value: str) -> str | None:
    """Return a problem string, or None if the value looks fine for its data_type."""
    if not value:
        return None
    dt = (data_type or "").lower()
    if dt == "cpf" and not CPF_RE.match(value):
        return "formato_cpf_invalido"
    if dt == "cpf":
        ok = cpf_check_digits_ok(value)
        if ok is False:
            return "digitos_verificadores_cpf_invalidos"
    if dt == "cep" and not CEP_RE.match(value):
        return "formato_cep_invalido"
    if dt == "uf" and value.strip().upper() not in UF_SET:
        return "uf_desconhecida"
    if dt == "date":
        if re.search(r"\bde\b.*\bde\b", value, flags=re.IGNORECASE):
            return None  # textual/"extenso" Portuguese date (e.g. "12 de novembro de 2006"); not numeric, nothing to check here
        import datetime as _dt
        for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d"):
            try:
                _dt.datetime.strptime(value, fmt)
                return None
            except ValueError:
                continue
        return "data_invalida"
    return None


def strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")


def normalize_loose(text: str) -> str:
    return re.sub(r"\s+", " ", strip_accents(text.lower())).strip()


def cer(hyp: str, ref: str) -> float | None:
    if not ref:
        return None
    if not hyp:
        return 1.0
    a, b = hyp, ref
    dp = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, len(b) + 1):
            cur = dp[j]
            dp[j] = prev if a[i - 1] == b[j - 1] else 1 + min(prev, dp[j], dp[j - 1])
            prev = cur
    return dp[len(b)] / len(b)


# ---------------------------------------------------------------------------
# Image/placement metrics
# ---------------------------------------------------------------------------

@dataclass
class InkMetrics:
    has_ink: bool
    # left_margin_h: horizontal offset of the ink's left edge from the crop's left edge,
    # expressed as a fraction of the FIELD HEIGHT (not width). This is the unit that
    # matters for "does the text start right after the printed rule/label", since form
    # fields are usually much wider than tall and a margin expressed as a fraction of
    # width would be dominated by field width rather than by how close the ink hugs the
    # left edge. Values >1 mean the ink starts more than one field-height to the right of
    # the left edge.
    left_margin_h: float | None = None
    # left_margin_w: the same offset, but as a fraction of field WIDTH (0..1 for ink that
    # starts inside the field). Kept alongside left_margin_h because composite digit-box
    # fields are often nearly square (height ~= width), where the two coincide, while wide
    # text fields differ a lot between the two normalizations; both are reported so a
    # reader is never stuck with a single, easily-misread fraction (see verification.md).
    left_margin_w: float | None = None
    right_overflow: bool | None = None
    vcenter_offset_frac: float | None = None
    ink_height_ratio: float | None = None
    hfill_ratio: float | None = None
    max_gap_over_median: float | None = None
    max_gap_over_ink_height: float | None = None
    excessive_spacing: bool | None = None


def ink_mask(crop: np.ndarray, blank_crop: np.ndarray | None) -> np.ndarray:
    gray = crop.mean(axis=2) if crop.ndim == 3 else crop.astype(float)
    if blank_crop is not None and blank_crop.shape == crop.shape[:2] + (crop.shape[2],) if crop.ndim == 3 else blank_crop is not None:
        bgray = blank_crop.mean(axis=2) if blank_crop.ndim == 3 else blank_crop.astype(float)
        if bgray.shape == gray.shape:
            diff = bgray - gray  # darker than the empty template = ink
            return diff > 28
    return gray < 200


def analyze_ink(crop: np.ndarray, blank_crop: np.ndarray | None, field_h_px: float, field_w_px: float | None = None) -> InkMetrics:
    mask = ink_mask(crop, blank_crop)
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return InkMetrics(has_ink=False)
    h, w = mask.shape
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    ink_h = y1 - y0 + 1
    ink_w = x1 - x0 + 1
    left_margin_h = x0 / field_h_px if field_h_px else None
    left_margin_w = x0 / field_w_px if field_w_px else None
    right_overflow = bool(x1 >= w - 2)
    vcenter = (y0 + y1) / 2
    vcenter_offset_frac = (vcenter - h / 2) / field_h_px if field_h_px else None
    ink_height_ratio = ink_h / h if h else None
    hfill_ratio = ink_w / w if w else None

    # column-cluster gap analysis (word/chunk spacing)
    col_has_ink = mask.any(axis=0)
    gaps = []
    run = 0
    in_gap = False
    cols = np.where(col_has_ink)[0]
    if cols.size > 1:
        runs = []
        start = cols[0]
        prev = cols[0]
        for c in cols[1:]:
            if c - prev > 1:
                runs.append((start, prev))
                start = c
            prev = c
        runs.append((start, prev))
        gaps = [runs[i + 1][0] - runs[i][1] for i in range(len(runs) - 1)]
    max_gap_over_median = None
    max_gap_over_ink_height = None
    excessive = None
    if gaps:
        med = statistics.median(gaps)
        max_gap_over_median = (max(gaps) / med) if med else None
        max_gap_over_ink_height = max(gaps) / ink_h if ink_h else None
        excessive = bool(max_gap_over_ink_height and max_gap_over_ink_height > 1.5)
    return InkMetrics(True, left_margin_h, left_margin_w, right_overflow, vcenter_offset_frac, ink_height_ratio, hfill_ratio,
                       max_gap_over_median, max_gap_over_ink_height, excessive)


def checkbox_mark_metrics(page_img: Image.Image, blank_img: Image.Image | None, element_bbox: list[float]) -> dict:
    """Per-element checkbox mark geometry: size vs box, center offset, a rough stroke-width
    estimate, dominant stroke angle (via PCA on ink pixel coordinates), and ink outside the
    box. Computed on a small margin around just the ONE checked element's bbox (not the
    whole field, which would include every unchecked option and its label)."""
    crop, _ = crop_field(page_img, element_bbox, margin=3)
    blank_crop = None
    if blank_img is not None:
        blank_crop, _ = crop_field(blank_img, element_bbox, margin=3)
    arr = np.array(crop)
    barr = np.array(blank_crop) if blank_crop is not None else None
    mask = ink_mask(arr, barr)
    ys, xs = np.nonzero(mask)
    box_w, box_h = element_bbox[2] - element_bbox[0], element_bbox[3] - element_bbox[1]
    if xs.size == 0:
        return {"mark_present": False}
    h, w = mask.shape
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    mark_w, mark_h = x1 - x0 + 1, y1 - y0 + 1
    size_ratio = (mark_w * mark_h) / (box_w * box_h) if box_w and box_h else None
    center_x, center_y = (x0 + x1) / 2, (y0 + y1) / 2
    # "inside the box" region of the crop (excluding the fixed margin added by crop_field)
    margin = 3
    inside = (xs >= margin) & (xs < w - margin) & (ys >= margin) & (ys < h - margin)
    ink_outside_frac = 1.0 - (inside.sum() / xs.size) if xs.size else None
    center_offset_frac = (math.hypot(center_x - w / 2, center_y - h / 2) / max(box_w, box_h)) if box_w and box_h else None
    # rough stroke-width estimate: ink area / the mark's longer dimension approximates the
    # average cross-section thickness of a single stroke (X or check mark)
    stroke_width_px = (xs.size / max(mark_w, mark_h)) if max(mark_w, mark_h) else None
    # dominant stroke angle via PCA on ink pixel coordinates
    angle_deg = None
    if xs.size >= 4:
        pts = np.stack([xs - center_x, ys - center_y]).astype(float)
        cov = np.cov(pts)
        if cov.shape == (2, 2) and np.all(np.isfinite(cov)):
            eigvals, eigvecs = np.linalg.eigh(cov)
            principal = eigvecs[:, int(np.argmax(eigvals))]
            angle_deg = round(math.degrees(math.atan2(principal[1], principal[0])) % 180, 1)
    return {
        "mark_present": True, "mark_size_ratio": round(size_ratio, 3) if size_ratio is not None else None,
        "mark_center_offset_frac": round(center_offset_frac, 3) if center_offset_frac is not None else None,
        "mark_stroke_width_px": round(stroke_width_px, 2) if stroke_width_px is not None else None,
        "mark_angle_deg": angle_deg,
        "mark_ink_outside_frac": round(ink_outside_frac, 3) if ink_outside_frac is not None else None,
    }


def run_tesseract(img: Image.Image, digits_only: bool = False, psm: int = 7) -> str:
    args = [TESSERACT_BIN, "stdin", "stdout", "-l", "por", "--psm", str(psm)]
    if digits_only:
        args += ["-c", "tessedit_char_whitelist=0123456789/-.,"]
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    try:
        out = subprocess.run(args, input=buf.getvalue(), capture_output=True, timeout=15)
        return out.stdout.decode("utf-8", "ignore").strip()
    except Exception:
        return ""


def run_tesseract_composite(page_img: Image.Image, elements: list[dict], digits_only: bool) -> str:
    """OCR a composite digit/letter box (e.g. a date or CPF rendered as one glyph per
    printed cell) one ELEMENT at a time with psm 10 (single character), instead of psm 7
    over the whole field crop (which mixes in the separator rules/box borders between
    cells and was responsible for 15-27% EMPTY tesseract reads on digit rows -- see
    verification.md). Falls back to "" per cell on OCR failure; cells are joined in
    element order, which already includes any literal separators recorded as elements."""
    chars = []
    for element in elements:
        bbox = element.get("bbox")
        if not bbox:
            continue
        crop, _ = crop_field(page_img, bbox, margin=2)
        text = run_tesseract(crop, digits_only=digits_only, psm=10).strip()
        chars.append(text)
    return "".join(chars)


# ---------------------------------------------------------------------------
# Core audit
# ---------------------------------------------------------------------------

@dataclass
class Row:
    document: str
    preset: str
    seed: int
    field_id: str
    label: str
    data_type: str
    selected: bool
    rendered: bool
    blank: bool
    blank_reason: str | None
    value: str
    error: str | None
    has_ink: bool | None = None
    left_margin_h: float | None = None
    left_margin_w: float | None = None
    right_overflow: bool | None = None
    vcenter_offset_frac: float | None = None
    ink_height_ratio: float | None = None
    hfill_ratio: float | None = None
    digit_height_ratio: float | None = None
    is_digit_field: bool = False
    max_gap_over_median: float | None = None
    max_gap_over_ink_height: float | None = None
    excessive_spacing: bool | None = None
    ocr_text: str | None = None
    cer_strict: float | None = None
    cer_loose: float | None = None
    data_problem: str | None = None
    is_checkbox: bool = False
    checkbox_checked_count: int | None = None
    field_shape: str | None = None  # "single_line" | "composite_box" | "checkbox" | "multiline"
    element_count: int = 0
    mark_present: bool | None = None
    mark_size_ratio: float | None = None
    mark_center_offset_frac: float | None = None
    mark_stroke_width_px: float | None = None
    mark_angle_deg: float | None = None
    mark_ink_outside_frac: float | None = None
    ai_eligible: bool = False
    timing_s: float | None = None
    crop_path: str | None = None
    content_class: str | None = None
    rendering_backend: str | None = None  # "font" | neural model name, from entry["rendering"]["backend"]
    rendering_fallback_from: str | None = None  # neural model name it fell back FROM, if any
    requested_value: str | None = None  # entry["requested_value"] when it differs from the rendered value
    ocr_text_easyocr: str | None = None
    cer_loose_easyocr: float | None = None
    ocr_engines_agree: bool | None = None  # both engines non-empty and loose-equal (digit fields only)


MULTILINE_DATA_TYPES = {"address", "document"}
MULTILINE_HEIGHT_PX_THRESHOLD = 100  # heuristic at ~300dpi; documented as approximate, see report.md


def classify_field_shape(data_type: str, is_checkbox: bool, element_count: int, bbox: list[float]) -> str:
    if is_checkbox:
        return "checkbox"
    if element_count >= 2:
        return "composite_box"
    height_px = bbox[3] - bbox[1] if bbox else 0
    if data_type in MULTILINE_DATA_TYPES and height_px > MULTILINE_HEIGHT_PX_THRESHOLD:
        return "multiline"
    return "single_line"


def _bbox_overlap_frac(a: list[float], b: list[float]) -> float:
    """Overlap area as a fraction of the SMALLER of the two bboxes' areas (so a small
    field fully inside a big one reports ~1.0, not a tiny fraction of the big one)."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    smaller = min(area_a, area_b) or 1.0
    return inter / smaller


def check_mapping_overlaps(definition: dict, threshold: float = 0.1) -> list[dict]:
    """Report fields of the same document whose bbox contains/overlaps another field's
    bbox by more than `threshold` of the smaller field's area (e.g. certidao_obito's
    data_obito_extenso bbox containing dia_obito's bbox). Reporting only -- mappings are
    never edited by the audit tool (see verification.md item 7)."""
    fields = [f for f in definition.get("fields", []) if f.get("bbox") and not f.get("unsupported_reason")]
    findings = []
    for i, f1 in enumerate(fields):
        for f2 in fields[i + 1:]:
            frac = _bbox_overlap_frac(f1["bbox"], f2["bbox"])
            if frac >= threshold:
                findings.append({"field_a": str(f1["field_id"]), "field_b": str(f2["field_id"]), "overlap_frac": round(frac, 3)})
    return findings


def selectable_fields(definition: dict) -> list[dict]:
    return [f for f in definition.get("fields", []) if not f.get("unsupported_reason")]


def render_page_png(pdf_path: Path, dpi: float) -> Image.Image:
    doc = fitz.open(pdf_path)
    page = doc[0]
    pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72.0, dpi / 72.0))
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    doc.close()
    return img


def crop_field(img: Image.Image, bbox: list[float], margin: int = 6) -> tuple[Image.Image, tuple[int, int, int, int]]:
    x0, y0, x1, y1 = bbox
    box = (max(0, int(x0) - margin), max(0, int(y0) - margin), min(img.width, int(x1) + margin), min(img.height, int(y1) + margin))
    return img.crop(box), box


def make_blank_template_png(template_path: Path, dpi: float) -> Image.Image:
    doc = fitz.open(template_path)
    if len(doc) > 1:
        doc.select([0])
    page = doc[0]
    pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72.0, dpi / 72.0))
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    doc.close()
    return img


def run_one(document_key: str, document, definition: dict, preset: dict, seed: int,
            field_filter: set[str] | None, out_dir: Path, blank_pages: dict[int, Image.Image],
            coherence_acc: dict | None = None, leakage_acc: dict | None = None,
            use_easyocr: bool = True) -> tuple[list[Row], float]:
    dpi = float(definition.get("dpi", 300))
    fields = selectable_fields(definition)
    if field_filter is not None:
        fields = [f for f in fields if str(f["field_id"]) in field_filter]
    selected_ids = {str(f["field_id"]) for f in fields}
    ai_keys, _fallback = ai_coverage(fields, document_key, dpi)

    rng = random.Random(f"{document_key}:{preset['uid']}:{seed}")
    pen = make_pen(preset)
    content_fonts: dict[str, str] = {}
    content_pens: dict[str, "NeuralPen | None"] = {}
    content_intensity: dict[str, float] = {}
    for content_class, override in preset.get("content", {}).items():
        model = override.get("model", "font")
        content_fonts[content_class] = override.get("font_family", preset["font_family"])
        content_intensity[content_class] = override.get("intensity", preset["intensity"])
        if model == "font" or NeuralPen is None:
            content_pens[content_class] = None
        else:
            content_pens[content_class] = NeuralPen(
                model=model, writer=override.get("writer"), bias=override.get("bias", 0.75),
                color=override.get("color", "#193787"), width=override.get("width", 1.5),
                pressure=override.get("pressure", 0.25), dropout=override.get("dropout", 0.0),
                text_policy=override.get("text_policy", "normalize"),
            )
    tmp_dir = out_dir / "_gen" / document_key / preset["uid"] / str(seed)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    rows: list[Row] = []
    try:
        doc = generate_document(
            document.template, definition, selected_ids, tmp_dir, f"{document_key}_{seed}", rng,
            handwriting=True, handwriting_intensity=preset["intensity"], text_handwriting=True,
            neural_pen=pen, font_family=preset["font_family"], realistic_filters=False,
            content_pens=content_pens or None, content_fonts=content_fonts or None,
            content_intensity=content_intensity or None,
        )
    except Exception as exc:  # whole-document failure (e.g. unsupported field slipped through)
        elapsed = time.monotonic() - t0
        for f in fields:
            rows.append(Row(document_key, preset["uid"], seed, str(f["field_id"]), f.get("label", ""),
                             f.get("data_type", "text"), True, False, True, "document_error", "", str(exc),
                             ai_eligible=str(f["field_id"]) in ai_keys, timing_s=elapsed))
        return rows, elapsed
    elapsed = time.monotonic() - t0

    truth = json.loads(doc.json_path.read_text(encoding="utf-8"))
    truth_by_id = {e["field_id"]: e for e in truth.get("fields", [])}
    page_img = render_page_png(doc.pdf_path, dpi)
    blank_img = blank_pages.get(round(dpi))

    pages_dir = out_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)
    page_img.save(pages_dir / f"{document_key}__{preset['uid']}__{seed}.png")

    if coherence_acc is not None:
        issues = check_coherence(truth.get("semantic_record") or {})
        bucket = coherence_acc.setdefault(document_key, {})
        for code in issues:
            bucket[code] = bucket.get(code, 0) + 1
        coherence_acc.setdefault("_checked_documents", set()).add(document_key)

    all_fields_map = {str(f["field_id"]): f for f in definition.get("fields", [])}
    for f in definition.get("fields", []):
        fid = str(f["field_id"])
        if field_filter is not None and fid not in field_filter:
            continue
        data_type = f.get("data_type", "text")
        unsupported = f.get("unsupported_reason")
        if unsupported:
            rows.append(Row(document_key, preset["uid"], seed, fid, f.get("label", ""), data_type,
                             False, False, True, f"unsupported:{unsupported}", "", None,
                             ai_eligible=fid in ai_keys, timing_s=elapsed))
            continue
        entry = truth_by_id.get(fid)
        if entry is None:
            continue
        value = entry.get("value", "")
        blank = bool(entry.get("blank")) or not value
        reason = None
        if entry.get("render_condition"):
            reason = "render_when"
        elif blank and not reason:
            reason = "blank_probability_or_null_binding" if value == "" else None
        element_count = len(entry.get("elements") or [])
        row = Row(document_key, preset["uid"], seed, fid, f.get("label", ""), data_type, True,
                  not blank, blank, reason, value, None, ai_eligible=fid in ai_keys, timing_s=elapsed)
        row.is_checkbox = bool(entry.get("elements")) and data_type == "checkbox"
        row.element_count = element_count
        semantic_key = (entry.get("semantic") or {}).get("key")
        row.content_class = classify_content(data_type, semantic_key, value)
        row.field_shape = classify_field_shape(data_type, row.is_checkbox, element_count, entry.get("bbox") or f.get("bbox"))
        if row.is_checkbox:
            row.checkbox_checked_count = sum(1 for e in entry["elements"] if e.get("checked"))

        rendering = entry.get("rendering") or {}
        row.rendering_backend = rendering.get("backend")
        row.rendering_fallback_from = (rendering.get("fallback") or {}).get("from_backend")
        requested = entry.get("requested_value")
        if requested is not None and requested != value:
            row.requested_value = requested
        if entry.get("blank_reason") and not row.blank_reason:
            row.blank_reason = entry["blank_reason"]

        # Blank-template-diffing leakage check: for a blank (unrendered) field, the crop
        # should show ~0 ink once diffed against the blank template. Measured once per
        # (document, field_id) across the whole run, independent of preset/seed.
        if blank and leakage_acc is not None:
            key = (document_key, fid)
            if key not in leakage_acc.setdefault("_checked", set()):
                leakage_acc["_checked"].add(key)
                bbox = entry.get("bbox") or f.get("bbox")
                if bbox and blank_img is not None:
                    crop, _ = crop_field(page_img, bbox)
                    blank_crop, _ = crop_field(blank_img, bbox)
                    mask = ink_mask(np.array(crop), np.array(blank_crop))
                    coverage = float(mask.mean())
                    leakage_acc.setdefault(document_key, []).append({"field_id": fid, "ink_coverage": round(coverage, 4)})

        if not blank:
            # Fix for "extenso date field crop unioned with an adjacent digit-cell field's
            # bbox" (verification.md): always crop/OCR using this field's OWN geometry --
            # the union of its own element bboxes when it has elements, never the raw
            # field["bbox"], which in a handful of mappings geometrically overlaps a
            # sibling field's bbox (see check_mapping_overlaps/item 7) and would otherwise
            # pull that sibling's pixels into this field's crop.
            own_elements = entry.get("elements") or []
            if own_elements and not row.is_checkbox:
                xs0 = [e["bbox"][0] for e in own_elements if e.get("bbox")]
                ys0 = [e["bbox"][1] for e in own_elements if e.get("bbox")]
                xs1 = [e["bbox"][2] for e in own_elements if e.get("bbox")]
                ys1 = [e["bbox"][3] for e in own_elements if e.get("bbox")]
                bbox = [min(xs0), min(ys0), max(xs1), max(ys1)] if xs0 else entry["bbox"]
            else:
                bbox = entry["bbox"]
            field_h_px = bbox[3] - bbox[1]
            field_w_px = bbox[2] - bbox[0]
            if row.is_checkbox:
                checked = [e for e in entry["elements"] if e.get("checked")]
                if checked:
                    geo = checkbox_mark_metrics(page_img, blank_img, checked[0]["bbox"])
                    row.mark_present = geo.get("mark_present")
                    row.mark_size_ratio = geo.get("mark_size_ratio")
                    row.mark_center_offset_frac = geo.get("mark_center_offset_frac")
                    row.mark_stroke_width_px = geo.get("mark_stroke_width_px")
                    row.mark_angle_deg = geo.get("mark_angle_deg")
                    row.mark_ink_outside_frac = geo.get("mark_ink_outside_frac")
                crop, _ = crop_field(page_img, bbox)
                row.has_ink = row.mark_present
                # generic ink-placement columns are not meaningful for a multi-option
                # checkbox field bbox (it covers every option + label) and are left unset
            else:
                crop, _ = crop_field(page_img, bbox)
                blank_crop = None
                if blank_img is not None:
                    blank_crop, _ = crop_field(blank_img, bbox)
                metrics = analyze_ink(np.array(crop), np.array(blank_crop) if blank_crop is not None else None, field_h_px, field_w_px)
                row.has_ink = metrics.has_ink
                row.left_margin_h = metrics.left_margin_h
                row.left_margin_w = metrics.left_margin_w
                row.right_overflow = metrics.right_overflow
                row.vcenter_offset_frac = metrics.vcenter_offset_frac
                row.ink_height_ratio = metrics.ink_height_ratio
                row.hfill_ratio = metrics.hfill_ratio
                row.max_gap_over_median = metrics.max_gap_over_median
                row.max_gap_over_ink_height = metrics.max_gap_over_ink_height
                row.excessive_spacing = metrics.excessive_spacing
                row.is_digit_field = data_type in {"date", "cpf", "digit_sequence", "integer"} or bool(
                    value and value.replace(".", "").replace("-", "").replace("/", "").isdigit())
                if row.is_digit_field:
                    row.digit_height_ratio = metrics.ink_height_ratio
                digits_only = data_type in {"date", "cpf", "digit_sequence", "integer"}
                # Composite digit/letter boxes (one printed cell per character) are OCR'd
                # per cell at psm 10, not as a whole run at psm 7: psm 7 was reading the
                # separator rules between cells as part of the line and coming back EMPTY
                # on 15-27% of digit rows for every style (see verification.md / item 2).
                if row.field_shape == "composite_box" and own_elements:
                    ocr = run_tesseract_composite(page_img, own_elements, digits_only)
                else:
                    ocr = run_tesseract(crop, digits_only=digits_only)
                row.ocr_text = ocr
                row.cer_strict = cer(ocr, value)
                row.cer_loose = cer(normalize_loose(ocr), normalize_loose(value))
                # Second OCR engine (easyocr) for digit fields, on by default when
                # installed: the verdict for digit legibility must use both engines plus
                # geometry (digit_height_ratio), never tesseract CER alone -- a single
                # engine's empty read is at least as likely to be an OCR limitation as a
                # rendering defect.
                if row.is_digit_field and use_easyocr and easyocr_available():
                    ocr2 = run_easyocr(crop)
                    row.ocr_text_easyocr = ocr2
                    row.cer_loose_easyocr = cer(normalize_loose(ocr2), normalize_loose(value))
                    if ocr and ocr2:
                        row.ocr_engines_agree = normalize_loose(ocr) == normalize_loose(ocr2)
                row.data_problem = validate_value(data_type, value)
            crop_path = out_dir / "crops" / document_key / preset["uid"] / f"{seed}_{fid}.png"
            crop_path.parent.mkdir(parents=True, exist_ok=True)
            crop.save(crop_path)
            row.crop_path = str(crop_path.relative_to(out_dir))
        rows.append(row)
    return rows, elapsed


# ---------------------------------------------------------------------------
# Generic cross-field coherence checks (driven by semantic key name patterns and
# data_type, not tied to one document template)
# ---------------------------------------------------------------------------

import datetime as _dt_mod  # noqa: E402

# IBGE state numeric-code prefixes (first two digits of a 7-digit municipality code),
# used for a generic municipality-code vs UF cross-check.
IBGE_UF_PREFIX = {
    "11": "RO", "12": "AC", "13": "AM", "14": "RR", "15": "PA", "16": "AP", "17": "TO",
    "21": "MA", "22": "PI", "23": "CE", "24": "RN", "25": "PB", "26": "PE", "27": "AL",
    "28": "SE", "29": "BA", "31": "MG", "32": "ES", "33": "RJ", "35": "SP", "41": "PR",
    "42": "SC", "43": "RS", "50": "MS", "51": "MT", "52": "GO", "53": "DF",
}


def _parse_any_date(value) -> "_dt_mod.date | None":
    if not isinstance(value, str) or not value.strip():
        return None
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y"):
        try:
            return _dt_mod.datetime.strptime(value.strip(), fmt).date()
        except ValueError:
            continue
    return None


def _implied_age(ref: "_dt_mod.date", birth: "_dt_mod.date") -> int:
    years = ref.year - birth.year
    if (ref.month, ref.day) < (birth.month, birth.day):
        years -= 1
    return years


def check_coherence(snapshot: dict) -> list[str]:
    """Generic cross-field coherence checks over a document's flat semantic snapshot
    (`ground_truth["semantic_record"]`, dotted English keys like "mother.birth_date",
    "deceased.age", "newborn.sex" -- the actual convention this generator uses). Driven by
    key-name tokens and value shape, not by any single document's layout, so the same
    function runs unmodified across all 17 documents. Returns a list of short problem
    codes (possibly repeated) found in this one record."""
    issues: list[str] = []
    today = _dt_mod.date.today()

    dates: dict[str, _dt_mod.date] = {}
    for key, value in snapshot.items():
        if not key.endswith(("date", "_date")):
            continue
        d = _parse_any_date(value)
        if d is not None:
            dates[key] = d
            if d > today:
                issues.append("data_no_futuro")

    personal_birth_keys = [k for k in dates if k.endswith("birth_date")]
    other_event_keys = [k for k in dates if not k.endswith("birth_date")]

    # a person's own birth date must precede every other date in the same document
    # (parents born before their child, spouses before their marriage, the deceased
    # before their own death/registration, etc.)
    for bk in personal_birth_keys:
        for ok in other_event_keys:
            if dates[ok] < dates[bk]:
                issues.append("data_anterior_ao_nascimento_da_pessoa")

    # an occurrence date (birth.date / death.date / marriage.date) should not be AFTER
    # its own registration/issuance date
    occurrence_keys = [k for k in other_event_keys if k in ("birth.date", "death.date", "marriage.date")]
    registration_keys = [k for k in other_event_keys
                          if "registration" in k or "issue_date" in k or k == "filling.date"]
    for ek in occurrence_keys:
        for rk in registration_keys:
            if dates[rk] < dates[ek]:
                issues.append("registro_anterior_ao_evento")

    # age vs birth date, referenced against the document's own central event date (NOT
    # "today" -- these are historical records) when one is unambiguously identifiable
    ref_date = next((dates[k] for k in ("death.date", "marriage.date", "birth.date") if k in dates), None)
    for key, value in snapshot.items():
        if not key.endswith(".age"):
            continue
        try:
            age_value = int(str(value).strip())
        except (TypeError, ValueError):
            continue
        prefix = key[: -len(".age")]
        birth_key = f"{prefix}.birth_date"
        if birth_key not in dates:
            continue
        reference = ref_date if prefix not in ("mother", "father") else dates.get("birth.date", ref_date)
        if reference is None:
            continue
        implied = _implied_age(reference, dates[birth_key])
        if abs(implied - age_value) > 1:
            issues.append("idade_inconsistente_com_data_nascimento")

    # mother's age plausibility (absolute range, independent of exact reference date)
    mother_age = snapshot.get("mother.age")
    try:
        if mother_age is not None and not (10 <= int(str(mother_age).strip()) <= 60):
            issues.append("idade_mae_implausivel")
    except (TypeError, ValueError):
        pass

    # sex vs gendered free text (loose heuristic: flag an explicit participle of the
    # opposite gender appearing anywhere else in the same record's free text)
    sex_value = next((str(v).strip().upper() for k, v in snapshot.items() if k.endswith(".sex") and v), None)
    if sex_value == "M":
        for key, value in snapshot.items():
            if isinstance(value, str) and re.search(r"\bnascida\b|\bfalecida\b", value, re.IGNORECASE):
                issues.append("sexo_inconsistente_com_texto")
    elif sex_value == "F":
        for key, value in snapshot.items():
            if isinstance(value, str) and re.search(r"\bnascido\b|\bfalecido\b", value, re.IGNORECASE):
                issues.append("sexo_inconsistente_com_texto")

    # municipality (IBGE code) vs UF, generic via the IBGE state-prefix table
    for key, value in snapshot.items():
        if not key.endswith(".ibge") or not value:
            continue
        code = str(value).strip()
        if len(code) < 2 or not code[:2].isdigit():
            continue
        prefix = key[: -len(".ibge")]
        uf_value = snapshot.get(f"{prefix}.uf")
        expected_uf = IBGE_UF_PREFIX.get(code[:2])
        if uf_value and expected_uf and str(uf_value).strip().upper() != expected_uf:
            issues.append("municipio_uf_inconsistente")

    # repeated person names across different roles in the same document (e.g. the
    # declarant sharing a name with a parent/spouse they are not supposed to be)
    name_by_key = {k: v for k, v in snapshot.items() if isinstance(v, str) and v.strip() and k.endswith(".name")}
    roles = {re.sub(r"\.name$", "", k): v for k, v in name_by_key.items()}
    values_seen: dict[str, str] = {}
    for role, value in roles.items():
        norm = normalize_loose(value)
        if len(norm) < 6:
            continue
        if norm in values_seen and values_seen[norm] != role:
            issues.append("nome_repetido_entre_papeis")
        values_seen[norm] = role

    return issues


SHEET_COLS = 4
SHEET_CELL_W, SHEET_CELL_H = 260, 110
SHEET_ROWS_PER_PAGE = 6  # 4x6 = 24 crops/page; paginate instead of cutting fields off


def write_sheet(out_dir: Path, document_key: str, preset_uid: str, rows: list[Row], seeds: list[int]) -> None:
    rows = [r for r in rows if r.document == document_key and r.preset == preset_uid and r.crop_path]
    if not rows:
        return
    per_page = SHEET_COLS * SHEET_ROWS_PER_PAGE
    sheets_dir = out_dir / "sheets"
    sheets_dir.mkdir(parents=True, exist_ok=True)
    n_pages = max(1, math.ceil(len(rows) / per_page))
    for page in range(n_pages):
        page_rows = rows[page * per_page:(page + 1) * per_page]
        grid_rows = max(1, math.ceil(len(page_rows) / SHEET_COLS))
        sheet = Image.new("RGB", (SHEET_COLS * SHEET_CELL_W, grid_rows * SHEET_CELL_H), "white")
        draw = ImageDraw.Draw(sheet)
        for i, r in enumerate(page_rows):
            cx, cy = (i % SHEET_COLS) * SHEET_CELL_W, (i // SHEET_COLS) * SHEET_CELL_H
            try:
                crop = Image.open(out_dir / r.crop_path)
                crop.thumbnail((SHEET_CELL_W - 10, SHEET_CELL_H - 32))
                sheet.paste(crop, (cx + 5, cy + 5))
            except Exception:
                pass
            caption = f"{r.field_id} s{r.seed} CER={r.cer_loose:.2f}" if r.cer_loose is not None else f"{r.field_id} s{r.seed}"
            draw.text((cx + 5, cy + SHEET_CELL_H - 24), caption[:40], fill="black")
            draw.text((cx + 5, cy + SHEET_CELL_H - 12), (r.value or "")[:40], fill="gray")
        suffix = "" if n_pages == 1 else f"__p{page + 1}of{n_pages}"
        sheet.save(sheets_dir / f"{document_key}__{preset_uid}{suffix}.png")


def load_rows_from_jsonl(path: Path) -> list[Row]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        data = json.loads(line)
        row = Row(**{k: data.get(k) for k in Row.__dataclass_fields__})
        # re-derive the data-plausibility check with the current (possibly fixed) validator
        if row.value and not row.is_checkbox:
            row.data_problem = validate_value(row.data_type, row.value)
        rows.append(row)
    return rows


def load_timings_from_log(path: Path) -> list[dict]:
    timings = []
    pattern = re.compile(r"^(\S+)\s+(\S+)\s+seed=(\d+)\s+([\d.]+)s\s+fields=\d+")
    if not path.is_file():
        return timings
    for line in path.read_text(encoding="utf-8").splitlines():
        m = pattern.match(line.strip())
        if m:
            timings.append({"document": m.group(1), "preset": m.group(2), "seed": int(m.group(3)), "seconds": float(m.group(4))})
    return timings


def rerun_report_only(out_dir: Path) -> None:
    metrics_path = out_dir / "metrics.jsonl"
    if not metrics_path.is_file():
        print(f"No metrics.jsonl at {metrics_path}; cannot run --report-only", file=sys.stderr)
        sys.exit(1)
    rows = load_rows_from_jsonl(metrics_path)
    # Row.timing_s is populated per-row since this version of the tool; fall back to
    # parsing run.log only for older metrics.jsonl files that predate that change.
    if any(r.timing_s is not None for r in rows):
        timings = [{"document": r.document, "preset": r.preset, "seed": r.seed, "seconds": r.timing_s}
                   for r in rows if r.timing_s is not None]
    else:
        timings = load_timings_from_log(out_dir / "run.log")
    doc_keys = sorted({r.document for r in rows})
    preset_uids = sorted({r.preset for r in rows})
    # Fix for item 6: a preset's "kind" must come from the same place build_presets()
    # computes it (standard font presets, content-style presets with no neural override
    # anywhere, and only THEN "neural") -- not from "is it one of STANDARD_PRESETS",
    # which mislabelled every content-style preset (e.g. padrao_fonte_* content variants)
    # as "neural" even when every one of its content overrides used model == "font".
    models_avail = neural_models_available()
    full_presets_by_uid = {p["uid"]: p for p in build_presets(models_avail)}
    kind_by_uid = {}
    for uid in preset_uids:
        if uid in full_presets_by_uid:
            kind_by_uid[uid] = full_presets_by_uid[uid]["kind"]
        else:
            kind_by_uid[uid] = "standard" if any(uid == p[0] for p in STANDARD_PRESETS) else "neural"
    presets = [{"uid": uid, "kind": kind_by_uid[uid]} for uid in preset_uids]
    # rewrite metrics.jsonl with the refreshed data_problem field, for diffability with future --compare runs
    with metrics_path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r.__dict__, ensure_ascii=False) + "\n")
    coherence_report = json.loads((out_dir / "coherence.json").read_text()) if (out_dir / "coherence.json").is_file() else {}
    leakage_report = json.loads((out_dir / "template_leakage.json").read_text()) if (out_dir / "template_leakage.json").is_file() else {}
    overlap_report = json.loads((out_dir / "mapping_overlaps.json").read_text()) if (out_dir / "mapping_overlaps.json").is_file() else {}
    summary = build_summary(rows, timings, presets, doc_keys, models_avail, coherence_report, leakage_report, overlap_report)
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    auto_report = build_report(summary, rows)
    report_path = out_dir / "report.md"
    existing = report_path.read_text(encoding="utf-8") if report_path.is_file() else ""
    marker = "<!-- AUTO-GENERATED TABLES BELOW (regenerated by --report-only; hand-authored section above is preserved) -->"
    hand_authored = existing.split(marker)[0].rstrip() if marker in existing else ""
    report_path.write_text((hand_authored + ("\n\n" if hand_authored else "") + marker + "\n\n" + auto_report), encoding="utf-8")
    print(f"Recomputed summary.json and the auto-generated part of report.md for {len(rows)} rows "
          f"({len(doc_keys)} documents x {len(presets)} presets) without regenerating any document.")


QUICK_DOCUMENTS = ["do", "dnv", "certidao_obito", "cnpj_quadro_socios_administradores"]
QUICK_PRESETS = ["padrao_fonte_natural", "padrao_fonte_homemade_apple", "padrao_digitada_arimo",
                  "style_20260921_144107_d6344a", "style_20260921_140241_6d9869"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--documents", nargs="*", default=None)
    parser.add_argument("--presets", nargs="*", default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--neural-seeds", nargs="*", type=int, default=None)
    parser.add_argument("--fields", nargs="*", default=None)
    parser.add_argument("--compare", default=None)
    parser.add_argument("--report-only", action="store_true",
                         help="Recompute summary.json/report.md from an existing metrics.jsonl (no generation, no OCR).")
    parser.add_argument("--quick", action="store_true",
                         help=f"Deterministic fast subset for iteration: documents={QUICK_DOCUMENTS}, "
                              f"presets={QUICK_PRESETS}, seeds 1 (+2 for non-neural presets). "
                              "Any of --documents/--presets/--seeds overrides the corresponding quick default.")
    parser.add_argument("--no-easyocr", action="store_true",
                         help="Disable the easyocr second engine on digit fields (default: on when installed).")
    args = parser.parse_args()

    out_dir = ROOT / "var" / "audit" / args.label

    if args.report_only:
        rerun_report_only(out_dir)
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "crops").mkdir(exist_ok=True)
    (out_dir / "sheets").mkdir(exist_ok=True)
    (out_dir / "pages").mkdir(exist_ok=True)

    types = discover_document_types(DOCUMENTS_ROOT)
    if args.quick:
        doc_keys = args.documents or QUICK_DOCUMENTS
        preset_filter = args.presets or QUICK_PRESETS
    else:
        doc_keys = args.documents or sorted(types.keys())
        preset_filter = args.presets
    models_avail = neural_models_available()
    presets = build_presets(models_avail)
    if preset_filter:
        presets = [p for p in presets if p["uid"] in preset_filter]
    field_filter = set(args.fields) if args.fields else None
    seeds = args.seeds or DEFAULT_SEEDS
    neural_seeds = args.neural_seeds or NEURAL_SEEDS

    blank_cache: dict[tuple[str, int], Image.Image] = {}
    all_rows: list[Row] = []
    timings: list[dict] = []
    coherence_acc: dict = {}
    leakage_acc: dict = {}
    overlap_report: dict[str, list[dict]] = {}
    run_t0 = time.monotonic()
    use_easyocr = not args.no_easyocr

    for dkey in doc_keys:
        document = types[dkey]
        definition = apply_shared_rules(json.loads(document.active_mapping.read_text(encoding="utf-8")))
        dpi = round(float(definition.get("dpi", 300)))
        blank_pages = {dpi: blank_cache.setdefault((dkey, dpi), make_blank_template_png(document.template, dpi))}
        overlaps = check_mapping_overlaps(definition)
        if overlaps:
            overlap_report[dkey] = overlaps
        for preset in presets:
            # Rerun-into-the-same-label: remove this (document, preset)'s stale crops
            # before writing new ones, so a label can be rerun without ambiguity between
            # old and new evidence (see verification.md item 4).
            stale_crops = out_dir / "crops" / dkey / preset["uid"]
            if stale_crops.is_dir():
                for stale in stale_crops.glob("*.png"):
                    stale.unlink()
            for stale_sheet in (out_dir / "sheets").glob(f"{dkey}__{preset['uid']}*.png"):
                stale_sheet.unlink()
            use_seeds = neural_seeds if preset["kind"] == "neural" else seeds
            for seed in use_seeds:
                stale_page = out_dir / "pages" / f"{dkey}__{preset['uid']}__{seed}.png"
                if stale_page.is_file():
                    stale_page.unlink()
                try:
                    rows, elapsed = run_one(dkey, document, definition, preset, seed, field_filter, out_dir, blank_pages,
                                             coherence_acc=coherence_acc, leakage_acc=leakage_acc, use_easyocr=use_easyocr)
                except Exception as exc:
                    rows, elapsed = [], 0.0
                    print(f"[FATAL] {dkey} {preset['uid']} seed={seed}: {exc}", file=sys.stderr)
                all_rows.extend(rows)
                timings.append({"document": dkey, "preset": preset["uid"], "seed": seed, "seconds": elapsed})
                print(f"{dkey:45s} {preset['uid']:35s} seed={seed} {elapsed:.2f}s fields={len(rows)}")
        for preset in presets:
            write_sheet(out_dir, dkey, preset["uid"], all_rows, seeds)
    total_wall_s = time.monotonic() - run_t0

    # metrics.jsonl
    with (out_dir / "metrics.jsonl").open("w", encoding="utf-8") as fh:
        for r in all_rows:
            fh.write(json.dumps(r.__dict__, ensure_ascii=False) + "\n")

    coherence_report = {k: v for k, v in coherence_acc.items() if not k.startswith("_")}
    leakage_report = {k: v for k, v in leakage_acc.items() if not k.startswith("_")}
    (out_dir / "coherence.json").write_text(json.dumps(coherence_report, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "template_leakage.json").write_text(json.dumps(leakage_report, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "mapping_overlaps.json").write_text(json.dumps(overlap_report, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = build_summary(all_rows, timings, presets, doc_keys, models_avail, coherence_report, leakage_report, overlap_report)
    summary["total_wall_s"] = round(total_wall_s, 1)
    summary["quick_mode"] = bool(args.quick)
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "report.md").write_text(build_report(summary, all_rows), encoding="utf-8")
    print(f"\nDone. {len(all_rows)} field rows across {len(doc_keys)} documents x {len(presets)} presets. "
          f"Wall time {total_wall_s:.1f}s.")
    print(f"Output: {out_dir}")

    if args.compare:
        write_compare(out_dir, args.label, args.compare)


def build_summary(rows: list[Row], timings: list[dict], presets: list[dict], doc_keys: list[str],
                   models_avail: dict[str, bool], coherence_report: dict | None = None,
                   leakage_report: dict | None = None, overlap_report: dict | None = None) -> dict:
    def pct(n, d):
        return round(100.0 * n / d, 1) if d else None

    per_doc = {}
    for dkey in doc_keys:
        drows = [r for r in rows if r.document == dkey]
        mapped = len({r.field_id for r in drows})
        selectable = len({r.field_id for r in drows if r.selected})
        rendered = len({(r.preset, r.seed, r.field_id) for r in drows if r.rendered})
        total_combos = len({(r.preset, r.seed, r.field_id) for r in drows if r.selected})
        per_doc[dkey] = {
            "mapped_fields": mapped, "selectable_fields": selectable,
            "unsupported_fields": mapped - selectable,
            "fill_rate_over_selectable_pct": pct(rendered, total_combos),
        }

    per_preset = {}
    for p in presets:
        prows = [r for r in rows if r.preset == p["uid"]]
        # field-shape-aware split: composite digit/letter boxes and checkboxes are
        # excluded from the generic spacing/left-margin aggregates (they are reported
        # separately below) because both are structurally different from a single typed
        # or handwritten run of text -- see verification.md and report.md item 12.
        plain_rows = [r for r in prows if r.field_shape == "single_line" or r.field_shape == "multiline"]
        composite_rows = [r for r in prows if r.field_shape == "composite_box"]
        checkbox_rows = [r for r in prows if r.field_shape == "checkbox"]

        cer_vals = [r.cer_loose for r in prows if r.cer_loose is not None]
        digit_cer_vals = [r.cer_loose for r in prows if r.cer_loose is not None and r.is_digit_field]
        gaps = [r.max_gap_over_ink_height for r in plain_rows if r.max_gap_over_ink_height is not None]
        composite_gaps = [r.max_gap_over_ink_height for r in composite_rows if r.max_gap_over_ink_height is not None]
        left_h = [r.left_margin_h for r in plain_rows if r.left_margin_h is not None]
        left_w = [r.left_margin_w for r in plain_rows if r.left_margin_w is not None]
        composite_left_h = [r.left_margin_h for r in composite_rows if r.left_margin_h is not None]
        digit_h = [r.digit_height_ratio for r in prows if r.digit_height_ratio is not None]

        mark_size = [r.mark_size_ratio for r in checkbox_rows if r.mark_size_ratio is not None]
        mark_center = [r.mark_center_offset_frac for r in checkbox_rows if r.mark_center_offset_frac is not None]
        mark_stroke = [r.mark_stroke_width_px for r in checkbox_rows if r.mark_stroke_width_px is not None]
        mark_outside = [r.mark_ink_outside_frac for r in checkbox_rows if r.mark_ink_outside_frac is not None]

        per_preset[p["uid"]] = {
            "kind": p["kind"], "n_rows": len(prows),
            "mean_cer_loose": round(statistics.mean(cer_vals), 3) if cer_vals else None,
            "mean_cer_loose_digits_only": round(statistics.mean(digit_cer_vals), 3) if digit_cer_vals else None,
            "excessive_spacing_count": sum(1 for r in plain_rows if r.excessive_spacing),
            "excessive_spacing_count_composite_box_excluded_above": sum(1 for r in composite_rows if r.excessive_spacing),
            "mean_max_gap_over_ink_height": round(statistics.mean(gaps), 2) if gaps else None,
            "mean_max_gap_over_ink_height_composite_box": round(statistics.mean(composite_gaps), 2) if composite_gaps else None,
            "mean_left_margin_h": round(statistics.mean(left_h), 3) if left_h else None,
            "mean_left_margin_w": round(statistics.mean(left_w), 3) if left_w else None,
            "mean_left_margin_h_composite_box": round(statistics.mean(composite_left_h), 3) if composite_left_h else None,
            "mean_digit_height_ratio": round(statistics.mean(digit_h), 3) if digit_h else None,
            "right_overflow_count": sum(1 for r in plain_rows if r.right_overflow),
            "n_checkbox_rows": len(checkbox_rows),
            "checkbox_mean_mark_size_ratio": round(statistics.mean(mark_size), 3) if mark_size else None,
            "checkbox_mean_center_offset_frac": round(statistics.mean(mark_center), 3) if mark_center else None,
            "checkbox_mean_stroke_width_px": round(statistics.mean(mark_stroke), 2) if mark_stroke else None,
            "checkbox_mean_ink_outside_frac": round(statistics.mean(mark_outside), 3) if mark_outside else None,
            "checkbox_multi_or_zero_mark_count": sum(1 for r in checkbox_rows if r.checkbox_checked_count != 1),
        }
        by_class: dict[str, dict] = {}
        for content_class in sorted({r.content_class for r in prows if r.content_class}):
            class_rows = [r for r in prows if r.content_class == content_class]
            class_cer = [r.cer_loose for r in class_rows if r.cer_loose is not None]
            by_class[content_class] = {"n_rows": len(class_rows),
                                       "mean_cer_loose": round(statistics.mean(class_cer), 3) if class_cer else None}
        per_preset[p["uid"]]["by_content_class"] = by_class
        digit_cer_easyocr = [r.cer_loose_easyocr for r in prows if r.cer_loose_easyocr is not None]
        per_preset[p["uid"]]["mean_cer_loose_digits_only_easyocr"] = (
            round(statistics.mean(digit_cer_easyocr), 3) if digit_cer_easyocr else None)
        agree = [r.ocr_engines_agree for r in prows if r.ocr_engines_agree is not None]
        per_preset[p["uid"]]["digit_ocr_engine_agreement_rate"] = (
            round(sum(1 for a in agree if a) / len(agree), 3) if agree else None)
        fallback_rows = [r for r in prows if r.rendering_fallback_from]
        per_preset[p["uid"]]["rendering_fallback_count"] = len(fallback_rows)
        blank_with_reason = sum(1 for r in prows if r.blank and r.blank_reason)
        blank_without_reason = sum(1 for r in prows if r.blank and not r.blank_reason)
        per_preset[p["uid"]]["blank_with_reason_count"] = blank_with_reason
        per_preset[p["uid"]]["blank_without_reason_count"] = blank_without_reason

    data_problems = {}
    for r in rows:
        if r.data_problem:
            data_problems.setdefault(r.data_problem, 0)
            data_problems[r.data_problem] += 1

    errors = {}
    for r in rows:
        if r.error:
            errors.setdefault(r.error[:120], 0)
            errors[r.error[:120]] += 1

    unsupported_total = sum(1 for r in rows if r.blank_reason and str(r.blank_reason).startswith("unsupported"))

    leakage_summary = {}
    for dkey, entries in (leakage_report or {}).items():
        coverages = [e["ink_coverage"] for e in entries]
        leaked = [e for e in entries if e["ink_coverage"] > 0.01]
        leakage_summary[dkey] = {"n_blank_fields_checked": len(entries),
                                  "mean_ink_coverage": round(statistics.mean(coverages), 4) if coverages else None,
                                  "max_ink_coverage": round(max(coverages), 4) if coverages else None,
                                  "fields_with_leakage_gt_1pct": [e["field_id"] for e in leaked]}

    return {
        "label": "baseline", "n_rows": len(rows), "n_documents": len(doc_keys), "n_presets": len(presets),
        "neural_models_available": models_avail,
        "per_document": per_doc, "per_preset": per_preset,
        "data_problems": data_problems, "errors": errors,
        "unsupported_field_rows": unsupported_total,
        "coherence": coherence_report or {},
        "template_leakage": leakage_summary,
        "mapping_overlaps": overlap_report or {},
        "mean_timing_s_by_preset": {p["uid"]: round(statistics.mean([t["seconds"] for t in timings if t["preset"] == p["uid"]] or [0]), 3)
                                     for p in presets},
        "total_timing_s": round(sum(t["seconds"] for t in timings), 1),
    }


def build_report(summary: dict, rows: list[Row]) -> str:
    lines = ["# Writing audit report — baseline", "",
             f"Documents: {summary['n_documents']}  Presets: {summary['n_presets']}  Field rows: {summary['n_rows']}",
             f"Total generation time (sum of per-combo elapsed): {summary['total_timing_s']} s"
             + (f"  Wall time: {summary['total_wall_s']} s" if "total_wall_s" in summary else ""),
             f"Quick mode: {summary.get('quick_mode', False)}",
             f"Neural models available: {summary['neural_models_available']}", "",
             "## Thresholds used",
             "- excessive_spacing: max inter-chunk gap / ink height > 1.5 (single_line/multiline fields only; "
             "composite_box fields are reported separately, see per-preset table)",
             "- right_overflow: ink touches the crop's right edge (crop margin 6px around bbox)",
             "- CER (loose): Levenshtein edit distance / len(ground truth), case/accent/whitespace-normalized",
             "- left_margin_h = ink left offset / field HEIGHT; left_margin_w = ink left offset / field WIDTH "
             "(both reported; see verification.md for why a single 'left_margin_frac' was ambiguous)",
             "- field_shape: single_line | multiline | composite_box (>=2 printed element boxes, e.g. date/CPF "
             "digit cells) | checkbox -- composite_box and checkbox rows are excluded from the generic "
             "spacing/margin aggregates above",
             "", "## Per-document fill", ""]
    for dkey, info in summary["per_document"].items():
        lines.append(f"- {dkey}: {info['selectable_fields']}/{info['mapped_fields']} selectable "
                      f"({info['unsupported_fields']} unsupported), fill={info['fill_rate_over_selectable_pct']}%")
    lines.append("")
    lines.append("## Per-preset metrics (single_line/multiline fields unless noted)")
    for uid, info in summary["per_preset"].items():
        lines.append(f"- {uid} ({info['kind']}): n={info['n_rows']} CER_loose={info['mean_cer_loose']} "
                      f"CER_loose_digits_only={info['mean_cer_loose_digits_only']} "
                      f"excessive_spacing={info['excessive_spacing_count']} "
                      f"(composite_box excessive_spacing, excluded above: {info['excessive_spacing_count_composite_box_excluded_above']}) "
                      f"max_gap/ink_h={info['mean_max_gap_over_ink_height']} "
                      f"left_margin_h={info['mean_left_margin_h']} left_margin_w={info['mean_left_margin_w']} "
                      f"(composite_box left_margin_h={info['mean_left_margin_h_composite_box']}) "
                      f"digit_h_ratio={info['mean_digit_height_ratio']} right_overflow={info['right_overflow_count']}")
        lines.append(f"    digits (2nd engine) easyocr CER_loose={info.get('mean_cer_loose_digits_only_easyocr')} "
                      f"engine_agreement_rate={info.get('digit_ocr_engine_agreement_rate')} "
                      f"rendering_fallback_count={info.get('rendering_fallback_count')} "
                      f"blank_with_reason={info.get('blank_with_reason_count')} "
                      f"blank_without_reason={info.get('blank_without_reason_count')}")
        if info["n_checkbox_rows"]:
            lines.append(f"    checkbox geometry (n={info['n_checkbox_rows']}): "
                          f"mark_size_ratio={info['checkbox_mean_mark_size_ratio']} "
                          f"center_offset_frac={info['checkbox_mean_center_offset_frac']} "
                          f"stroke_width_px={info['checkbox_mean_stroke_width_px']} "
                          f"ink_outside_frac={info['checkbox_mean_ink_outside_frac']} "
                          f"multi_or_zero_mark={info['checkbox_multi_or_zero_mark_count']}")
        by_class = info.get("by_content_class") or {}
        if by_class:
            class_parts = ", ".join(f"{cls}: n={cinfo['n_rows']} CER_loose={cinfo['mean_cer_loose']}"
                                    for cls, cinfo in by_class.items())
            lines.append(f"    by content class -- {class_parts}")
    lines.append("")
    lines.append("## Data problems")
    for k, v in sorted(summary["data_problems"].items(), key=lambda kv: -kv[1]):
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## Errors")
    for k, v in sorted(summary["errors"].items(), key=lambda kv: -kv[1]):
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## Cross-field coherence check counts per document")
    lines.append("(0 for a document means the checks ran and found nothing, UNLESS the document never exercises "
                  "the relevant semantic keys -- e.g. a procuração has no birth/death dates to check)")
    for dkey, issues in summary.get("coherence", {}).items():
        if dkey.startswith("_"):
            continue
        if issues:
            lines.append(f"- {dkey}: " + ", ".join(f"{k}={v}" for k, v in sorted(issues.items(), key=lambda kv: -kv[1])))
        else:
            lines.append(f"- {dkey}: none found")
    lines.append("")
    lines.append("## Blank-template ink-diffing leakage check (all documents, one page each)")
    lines.append("For every field that came back blank in at least one combo, ink coverage of that field's bbox "
                  "is measured after diffing against the blank template render. >1% coverage on a supposedly-blank "
                  "field would mean the diffing is picking up pre-printed template content as 'ink' (leakage).")
    for dkey, info in summary.get("template_leakage", {}).items():
        flag = f" LEAKAGE in: {info['fields_with_leakage_gt_1pct']}" if info["fields_with_leakage_gt_1pct"] else ""
        lines.append(f"- {dkey}: checked {info['n_blank_fields_checked']} blank fields, "
                      f"mean coverage={info['mean_ink_coverage']}, max={info['max_ink_coverage']}{flag}")
    lines.append("")
    lines.append("## Mapping bbox overlaps (report-only; mappings are not edited by this tool)")
    overlaps = summary.get("mapping_overlaps") or {}
    if overlaps:
        for dkey, findings in overlaps.items():
            for f in findings:
                lines.append(f"- {dkey}: {f['field_a']} overlaps {f['field_b']} (overlap_frac={f['overlap_frac']})")
    else:
        lines.append("- none found")
    lines.append("")
    lines.append("(See the accompanying hand-authored ranked problem list in this report for severity x frequency analysis and crop evidence.)")
    return "\n".join(lines)


_COMPARE_SCALAR_METRICS = ("cer_loose", "left_margin_h", "max_gap_over_ink_height", "digit_height_ratio")


def _save_side_by_side(base_dir: Path, other_dir: Path, out_path: Path, row_other: dict, row_base: dict, caption: str) -> bool:
    """Paste the other-run and base-run crops for one field side by side with a caption,
    so --compare gives visual evidence for the fields with the biggest metric swings, not
    only compare.md numbers (item 5)."""
    try:
        img_o = Image.open(other_dir / row_other["crop_path"]).convert("RGB") if row_other.get("crop_path") else None
        img_b = Image.open(base_dir / row_base["crop_path"]).convert("RGB") if row_base.get("crop_path") else None
    except Exception:
        return False
    if img_o is None and img_b is None:
        return False
    w = max((img_o.width if img_o else 0), (img_b.width if img_b else 0), 200)
    h = max((img_o.height if img_o else 0), (img_b.height if img_b else 0))
    pad = 24
    canvas = Image.new("RGB", (w * 2 + 10, h + pad), "white")
    draw = ImageDraw.Draw(canvas)
    if img_o is not None:
        canvas.paste(img_o, (0, pad))
    if img_b is not None:
        canvas.paste(img_b, (w + 10, pad))
    draw.text((2, 2), f"before | after -- {caption}"[:90], fill="black")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    return True


def write_compare(out_dir: Path, label: str, other_label: str) -> None:
    base_dir = out_dir
    other_dir = base_dir.parent / other_label
    if not other_dir.is_dir():
        print(f"compare target {other_dir} not found", file=sys.stderr)
        return
    base_summary = json.loads((base_dir / "summary.json").read_text())
    other_summary = json.loads((other_dir / "summary.json").read_text())
    lines = [f"# Compare {other_label} -> {label}", ""]
    for uid in base_summary["per_preset"]:
        b, o = base_summary["per_preset"].get(uid, {}), other_summary["per_preset"].get(uid, {})
        if not o:
            continue
        lines.append(f"- {uid}: CER_loose {o.get('mean_cer_loose')} -> {b.get('mean_cer_loose')}, "
                      f"CER_loose_digits_only {o.get('mean_cer_loose_digits_only')} -> {b.get('mean_cer_loose_digits_only')}, "
                      f"excessive_spacing {o.get('excessive_spacing_count')} -> {b.get('excessive_spacing_count')}, "
                      f"left_margin_h {o.get('mean_left_margin_h')} -> {b.get('mean_left_margin_h')}")
    (base_dir / "compare.md").write_text("\n".join(lines), encoding="utf-8")

    # Per-field side-by-side images: the N fields with the largest metric change across
    # any of _COMPARE_SCALAR_METRICS, plus a random sample, matched by
    # (document, preset, seed, field_id) between the two runs' metrics.jsonl.
    base_metrics_path, other_metrics_path = base_dir / "metrics.jsonl", other_dir / "metrics.jsonl"
    if not (base_metrics_path.is_file() and other_metrics_path.is_file()):
        return

    def _load(path: Path) -> dict[tuple, dict]:
        out = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            out[(d["document"], d["preset"], d["seed"], d["field_id"])] = d
        return out

    base_rows, other_rows = _load(base_metrics_path), _load(other_metrics_path)
    deltas = []
    for key, rb in base_rows.items():
        ro = other_rows.get(key)
        if ro is None or not rb.get("crop_path") or not ro.get("crop_path"):
            continue
        max_delta = 0.0
        for metric in _COMPARE_SCALAR_METRICS:
            vb, vo = rb.get(metric), ro.get(metric)
            if isinstance(vb, (int, float)) and isinstance(vo, (int, float)):
                max_delta = max(max_delta, abs(vb - vo))
        deltas.append((max_delta, key))
    deltas.sort(key=lambda t: -t[0])
    top_n = [key for _, key in deltas[:20]]
    rng = random.Random(f"compare:{label}:{other_label}")
    remaining = [key for _, key in deltas[20:]]
    sample = rng.sample(remaining, min(10, len(remaining))) if remaining else []

    compare_dir = base_dir / "compare_images"
    for key in top_n + sample:
        document, preset, seed, field_id = key
        rb, ro = base_rows[key], other_rows[key]
        caption = f"{document}/{preset}/s{seed}/{field_id}"
        fname = f"{document}__{preset}__{seed}__{field_id}.png".replace("/", "_")
        _save_side_by_side(base_dir, other_dir, compare_dir / fname, ro, rb, caption)
    print(f"Wrote {len(top_n)} largest-delta + {len(sample)} random side-by-side compare images to {compare_dir}")


if __name__ == "__main__":
    main()
