from __future__ import annotations

import copy
import json
import io
import functools
import hashlib
import random
import re
import secrets
import string
import unicodedata
from dataclasses import asdict, dataclass, replace
from functools import partial
from datetime import date, time, timedelta
from pathlib import Path
from typing import Any, Callable

import fitz
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from src.content_classes import classify_content
from src.document_types import build_dataset_run_dir, infer_document_type_key
from src.document_filters import apply_realistic_filters_to_pdf
from src.field_rules import (DATE_PARTS, DNV_TEST_PREFIX, SEPARATED_DATE, SEX_VALUES, TEXT_TYPES, apply_shared_rules, dnv_metadata,
                              document_number, format_date, format_dnv, masked_value, sex_text)
from src.fonts import DEFAULT_FONT_FAMILY, font_path, is_typed
from src.neural_handwriting import NeuralPen, load_model, prepare_text, render_strokes, trim_trailing_stray_mark
from src.semantic import REGISTRY, VERSION as SEMANTIC_VERSION, SemanticRecord
from src.semantic.assisted import ai_binding_key, serialize_ai_values
from src.semantic.custom import generate_custom_value, validate_custom_definition
from src.semantic.data import CITIES, COMPLEMENTS, FIRST_NAMES, LAST_NAMES, LOCATION_PROFILES, OCCUPATIONS, STREETS, UF_VALUES
from src.semantic.plan import SemanticPlan, semantic_plan
from src.semantic.present import checkbox_choice, present_elements, present_text, validation_rules, value_kind
from src.semantic.registry import CONTEXTS
from src.semantic.resolve import Binding, document_context
from src.writing_presets import document_id_with_preset


ASSETS_ROOT = Path(__file__).resolve().parent.parent / "assets"
HANDWRITING_FONT = ASSETS_ROOT / "fonts" / "Caveat-Regular.ttf"
TYPED_FONT_ALIAS_PREFIX = "ocrsd-typed-"

# Typed-text (printed-font) placement tuning. Root cause of the issues this fixes:
# the old `insert_printed` capped font size at 11pt regardless of field height (digits
# rendered far smaller than the field, digit_height_ratio ~0.34-0.38 in the writing
# audit), aligned text flush with the box's left edge (left_margin_w ~0.018-0.02), and
# relied on PyMuPDF's `insert_textbox` top-left anchor with no vertical centring.
TYPED_TARGET_DIGIT_HEIGHT_FRAC = 0.60  # digits should fill ~55-70% of field height
TYPED_MIN_FONT_SIZE = 5.0
TYPED_MAX_FONT_SIZE = 42.0  # ceiling for single-character cells only (see below)
TYPED_GENERIC_MAX_SIZE = 12.0  # ceiling for ordinary (multi-character) typed fields.
# (Added after the writing-audit verifier found this: a "fill-in-the-line" field's box
# is often tall for layout/visual-grid reasons, not because a typed line is meant to
# fill it -- e.g. a CNPJ quadro-de-socios table cell or a CadUnico address line. Sizing
# purely from rect.height (as single-character digit/CPF cells correctly do, since
# those boxes ARE sized to one glyph) made such fields' text balloon to the old
# TYPED_MAX_FONT_SIZE, and because the target size depends only on geometry, not
# content, a short value that happened to fit at that huge size (e.g. "Socio") rendered
# much larger than a long value in the same column that had to shrink to fit (e.g.
# "Socio-Administrador") -- same box, wildly different apparent sizes. Real typewritten
# forms run about 8-12pt regardless of box height; capping multi-character fields to
# TYPED_GENERIC_MAX_SIZE keeps the whole column in that realistic, consistent range.
# Single-character cells keep TYPED_MAX_FONT_SIZE because their height-to-glyph
# relationship is the whole point of the earlier digit-size fix.)
TYPED_SHRINK_FLOOR = 4.5  # preferred legibility floor in points; shrink stops here first.
# (Tuned down from an initial 6.0pt: the writing audit showed 6.0pt left ~20 long
# single_line fields -- addresses, "BRASILEIRO(A) NATURALIZADO(A)"-style enum values --
# clipped mid-word because their natural shrink-to-fit size (~4.6-5.3pt) was just below
# the floor. 4.5pt is still far more legible than the old code's unbounded shrink, which
# went under 2pt for these same fields. If a value still doesn't fit at 4.5pt,
# insert_printed tries once more down to TYPED_HARD_FLOOR; if it STILL doesn't fit, the
# field is left blank with a `blank_reason` in the ground truth rather than ever drawing
# text that overflows or clips -- the rendered PDF text must always equal the ground
# truth, per the writing-audit hard rule.)
TYPED_SHRINK_STEP = 0.92
TYPED_HARD_FLOOR = 3.0  # absolute last-resort size, below TYPED_SHRINK_FLOOR; see
# insert_printed -- OCR training data must never show clipped text, so when even
# TYPED_SHRINK_FLOOR doesn't fit we try once more down to this hard floor before
# giving up and leaving the field blank-with-reason instead of drawing over the edge.
TYPED_LEFT_PAD_HEIGHT_FRAC = 0.18  # left padding, proportional to field height
TYPED_MIN_LEFT_PAD_PT = 1.4  # minimum left padding in points
TYPED_LEFT_PAD_MAX_WIDTH_FRAC = 0.15  # cap the height-proportional pad for narrow fields
TYPED_RIGHT_PAD_PT = 1.0

# Font-handwriting left-alignment tuning (render_handwritten_text). Baseline left
# margins for pen/pencil presets ran 0.12-0.21 width-fraction when text happened to be
# narrower than the field and get centred; we target the low end of that same range
# for a natural "person started writing near the edge" look, not literally flush.
HANDWRITING_LEFT_PAD_HEIGHT_FRAC = 0.12
HANDWRITING_MIN_LEFT_PAD_PX = 2


@functools.lru_cache(maxsize=16)
def _typed_digit_height_ratio(fontfile: str) -> float:
    """Ratio of a rendered digit's ink height to the font's point size, measured from
    the actual glyph outline (not the font's overall ascender/descender span, which is
    much taller than a digit and would under-size text). Falls back to a conservative
    default if the font file can't be read."""
    try:
        font = ImageFont.truetype(fontfile, 200)
        bbox = font.getbbox("0")
    except Exception:
        return 0.62
    if not bbox:
        return 0.62
    ratio = (bbox[3] - bbox[1]) / 200.0
    return max(0.3, min(1.0, ratio))


@functools.lru_cache(maxsize=16)
def _typed_font_face(fontfile: str | None, fontname: str) -> fitz.Font:
    return fitz.Font(fontfile=fontfile, fontname=None if fontfile else fontname)
NUMERIC_CHARACTERS = frozenset("0123456789.,:/- ")
LOCATION_PREFIXES = {"occurrence": ("birth.place", "death.place"), "residence": ("mother.residence", "deceased.residence"),
                     "registry": ("registration.office", "death_registration.office")}
OBSTETRIC_TERMS = (("mother.obstetric.previous_pregnancies", ("gestacoes anteriores",)),
                   ("mother.obstetric.vaginal_births", ("partos vaginais",)), ("mother.obstetric.cesareans", ("cesareas",)),
                   ("mother.obstetric.live_births", ("nascidos vivos",)), ("mother.obstetric.fetal_losses", ("perdas fetais", "abortos")))
DEFAULT_LINKS = {"sexo": {"declaracao_obito": "deceased.sex", None: "newborn.sex"}, "dnv": {None: "registration.dnv_number"}}


@dataclass(frozen=True)
class SyntheticDocument:
    pdf_path: Path
    json_path: Path
    values: dict[str, str]


@dataclass(frozen=True)
class HandwritingStyle:
    ink: tuple[int, int, int]
    rotation: float
    width_scale: float
    size_scale: float
    spacing_scale: float
    stroke_bias: int
    baseline_bias: float


def random_handwriting_style(rng: random.Random, intensity: float = 0.65,
                             medium: str = "pen") -> HandwritingStyle:
    intensity = min(1.0, max(0.0, intensity))
    inks = (
        ((62, 62, 66), (70, 68, 66), (54, 57, 60), (78, 75, 71))
        if medium == "pencil"
        else ((20, 45, 115), (25, 55, 135), (30, 45, 90), (35, 55, 105), (28, 38, 65))
    )
    return HandwritingStyle(
        ink=rng.choice(inks),
        rotation=rng.uniform(-2.2, 2.2) * intensity,
        width_scale=rng.uniform(0.90, 1.08),
        size_scale=rng.uniform(0.91, 1.03),
        spacing_scale=rng.uniform(0.88, 1.06),
        stroke_bias=rng.choice((-1, 0, 0, 0, 1)) if intensity else 0,
        baseline_bias=rng.uniform(-0.035, 0.035) * intensity,
    )


def is_numeric_text(text: str) -> bool:
    return bool(text) and all(character in NUMERIC_CHARACTERS for character in text)


def _normalized_text(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value)
    return "".join(character for character in decomposed if not unicodedata.combining(character)).lower()


def _location_group(field: dict) -> str | None:
    text = _normalized_text(" ".join(str(field.get(key, "")) for key in ("field_id", "label", "description")))
    if "cartorio" in text:
        return "registry"
    if "residencia" in text or "logradouro" in text:
        return "residence"
    if "ocorrencia" in text or "endereco da ocorrencia" in text:
        return "occurrence"
    return None


def _allowed(key: str, document: str | None) -> bool:
    return document not in CONTEXTS or document in REGISTRY[key].contexts


RecordFactory = Callable[[int, str | None, SemanticPlan, list[str]], SemanticRecord]


def _semantic_context(document: str | None, seed: int) -> dict:
    document = document_context(document)
    return {"record": SemanticRecord(seed, document), "document": document}


def _record(context: dict, rng: random.Random) -> SemanticRecord:
    if context.get("record") is None:
        context.update(_semantic_context(context.get("document"), rng.getrandbits(64)))
    return context["record"]


def _location_key(field: dict, context: dict) -> str | None:
    group = _location_group(field)
    if group is None:
        return None
    key = f'{LOCATION_PREFIXES[group][context.get("document") == "declaracao_obito"]}.municipality'
    return key if _allowed(key, context.get("document")) else None


def _location_profile(field: dict, rng: random.Random, context: dict) -> dict | None:
    key = _location_key(field, context)
    if key is None:
        return None
    locations, group = context.setdefault("locations", {}), _location_group(field)
    if group not in locations:
        city = _record(context, rng).value(key)
        locations[group] = dict(next(profile for profile in LOCATION_PROFILES if profile["city"] == city))
    return locations[group]


def _obstetric_key(field: dict, context: dict) -> str | None:
    label = _normalized_text(" ".join(str(field.get(name, "")) for name in ("field_id", "label", "description", "notes")))
    key = next((key for key, terms in OBSTETRIC_TERMS if any(term in label for term in terms)), None)
    return key if key and _allowed(key, context.get("document")) else None


def _legacy_keys(field: dict, context: dict) -> list[str]:
    return [key for key in (_location_key(field, context), _obstetric_key(field, context)) if key]


def _bounded_integer(field: dict, rng: random.Random, context: dict,
                     width: int | None = None) -> int:
    validation = validation_rules(field)
    label = _normalized_text(" ".join(str(field.get(key, "")) for key in ("field_id", "label", "description", "notes")))
    if key := _obstetric_key(field, context):
        return _record(context, rng).value(key)
    if "apgar" in label:
        return rng.randint(int(validation.get("min", 0)), int(validation.get("max", 10)))
    if "semanas de gestacao" in label:
        return rng.randint(22, 42)
    if "consultas" in label and "pre-natal" in label:
        return rng.randint(0, 14)
    if re.search(r"(?<![a-z0-9])idade(?![a-z0-9])", label):
        return rng.randint(18, 65)
    if "numero" in label or "número" in label:
        upper = min(999, 10 ** (width or 3) - 1)
        return rng.randint(1, upper)
    minimum = int(validation.get("min", validation.get("min_value", 0)))
    maximum = int(validation.get("max", validation.get("max_value", 10 ** (width or 2) - 1)))
    return rng.randint(minimum, maximum)


def _random_digit_sequence(field: dict, rng: random.Random, context: dict) -> str:
    elements = field.get("elements") or []
    width = len(elements) if elements else int(validation_rules(field).get("digit_count", 1))
    validation = validation_rules(field)
    label = _normalized_text(" ".join(str(field.get(key, "")) for key in ("field_id", "label", "description", "notes")))
    profile = _location_profile(field, rng, context)
    if profile and ("cep" in label or validation.get("pattern") == "^[0-9]{8}$"):
        return profile["cep"][:width]
    if profile and ("codigo (ibge)" in label or "municipio de ocorrencia - codigo" in label):
        return profile["ibge"][:width]
    if validation.get("format") == "DDMMYYYY" or "data" in label:
        generated = date(1980, 1, 1) + timedelta(days=rng.randrange(0, 46 * 365))
        date_digits = generated.strftime("%d%m%Y")
        return date_digits[:width]
    if "peso" in label:
        return f"{rng.randint(700, 999 if width <= 3 else 4999):0{width}d}"[-width:]
    if "comprimento" in label:
        return f"{rng.randint(450, 550):03d}"[:width]
    if "perimetro" in label:
        return f"{rng.randint(30, 38):02d}"[:width]
    value = _bounded_integer(field, rng, context, width)
    return f"{value:0{width}d}"[-width:]


def _ensure_matches_validation(value: str, field: dict, rng: random.Random,
                               context: dict | None = None) -> str:
    validation = validation_rules(field)
    pattern = validation.get("pattern")
    if not pattern or re.fullmatch(pattern, value):
        return value
    if pattern == "^[0-9]{8}$":
        return "".join(str(rng.randrange(10)) for _ in range(8))
    if validation.get("allow_digits") is False:
        return f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)} {rng.choice(LAST_NAMES)}"
    return value


_MT_BULK_MIN = 4096
_MT = np.random.MT19937()


def _uniform_array(rng: random.Random, a: float, b: float, shape: tuple[int, ...]) -> np.ndarray:
    """Return float64 values equal to `rng.uniform(a, b)` called in C order, and advance `rng` the same way.
    Large requests copy the Mersenne Twister state into numpy and back."""
    count = int(np.prod(shape))
    if (count < _MT_BULK_MIN or type(rng).random is not random.Random.random
            or type(rng).uniform is not random.Random.uniform or type(rng).getstate is not random.Random.getstate):
        uniform = rng.uniform
        return np.fromiter((uniform(a, b) for _ in range(count)), dtype=np.float64, count=count).reshape(shape)
    version, state, gauss = rng.getstate()
    _MT.state = {"bit_generator": "MT19937", "state": {"key": np.array(state[:624], dtype=np.uint32), "pos": state[624]}}
    raw = _MT.random_raw(2 * count).reshape(count, 2)
    values = ((raw[:, 0] >> 5).astype(np.float64) * 67108864.0 + (raw[:, 1] >> 6)) * (1.0 / 9007199254740992.0)
    after = _MT.state["state"]
    rng.setstate((version, (*after["key"].tolist(), int(after["pos"])), gauss))
    return (a + (b - a) * values).reshape(shape)


def _apply_ballpoint_texture(mask: np.ndarray, rng: random.Random, intensity: float) -> np.ndarray:
    if not intensity or mask.size == 0:
        return mask
    height, width = mask.shape


    knot_count = max(3, min(8, width // 12 + 2))
    pressure_knots = np.array(
        [rng.uniform(0.48, 1.0) for _ in range(knot_count)], dtype=np.float32,
    ).reshape(1, knot_count)
    pressure = cv2.resize(pressure_knots, (width, height), interpolation=cv2.INTER_CUBIC)


    rows, columns = max(2, height // 14), max(3, width // 14)
    patch_values = []
    for _row in range(rows):
        row = []
        for _column in range(columns):
            value = rng.uniform(0.68, 1.0)
            if rng.random() < 0.16 * intensity:
                value = rng.uniform(0.22, 0.55)
            row.append(value)
        patch_values.append(row)
    patches = cv2.resize(
        np.asarray(patch_values, dtype=np.float32), (width, height),
        interpolation=cv2.INTER_CUBIC,
    )


    fine = _uniform_array(rng, 0.78, 1.0, (height, width)).astype(np.float32)
    fine = cv2.GaussianBlur(fine, (0, 0), sigmaX=0.55, sigmaY=0.55)
    texture = np.clip(pressure * patches * fine, 0.12, 1.0)
    texture /= max(float(texture.max()), 0.01)
    strength = 0.35 + 0.65 * intensity
    texture = 1.0 - (1.0 - texture) * strength
    return np.clip(mask.astype(np.float32) * texture, 0, 255).astype(np.uint8)


def _apply_pencil_texture(mask: np.ndarray, rng: random.Random, intensity: float) -> np.ndarray:
    if mask.size == 0:
        return mask
    height, width = mask.shape
    grain = _uniform_array(rng, 0.48, 1.0, (height, width)).astype(np.float32)
    grain = cv2.GaussianBlur(grain, (0, 0), sigmaX=0.28, sigmaY=0.28)
    pressure_knots = np.asarray(
        [rng.uniform(0.62, 1.0) for _ in range(max(3, width // 10 + 2))], dtype=np.float32,
    ).reshape(1, -1)
    pressure = cv2.resize(pressure_knots, (width, height), interpolation=cv2.INTER_CUBIC)
    texture = np.clip(grain * pressure, 0.25, 1.0)
    strength = 0.55 + 0.35 * min(1.0, max(0.0, intensity))
    texture = 1.0 - (1.0 - texture) * strength
    graphite = mask.astype(np.float32) * texture * rng.uniform(0.72, 0.88)
    return np.clip(graphite, 0, 255).astype(np.uint8)


@functools.lru_cache(maxsize=64)
def _handwriting_font(fontfile: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(fontfile, size)


@functools.lru_cache(maxsize=512)
def _glyph_outline(character: str, height: int, font_size: int, stroke: int, fontfile: str) -> Image.Image | None:
    """Return the cropped 4x glyph mask. Callers must not change the image in place."""
    scale = 4
    font = _handwriting_font(fontfile, font_size)
    bbox = font.getbbox(character or " ", stroke_width=1)
    width = max(round(height * scale * 0.20), bbox[2] - bbox[0] + 10 * scale)
    glyph = Image.new("L", (width, height * scale + 12 * scale), 0)
    ImageDraw.Draw(glyph).text(
        (glyph.width // 2, glyph.height // 2), character, font=font, fill=235,
        anchor="mm", stroke_width=stroke, stroke_fill=220,
    )
    crop = glyph.getbbox()
    if crop is None:
        return None
    return glyph.crop((max(0, crop[0]-4), max(0, crop[1]-4), min(glyph.width, crop[2]+4), min(glyph.height, crop[3]+4)))


def _render_glyph(character: str, height: int, style: HandwritingStyle,
                  rng: random.Random, intensity: float, medium: str = "pen",
                  font_family: str = DEFAULT_FONT_FAMILY) -> Image.Image:
    scale = 4
    glyph = _glyph_outline(character, height, max(12, round(height * scale * 1.08 * style.size_scale)),
                           max(0, 1 + style.stroke_bias), str(font_path(font_family)))
    if glyph is None:
        return Image.new("L", (max(1, round(height * 0.3)), height), 0)
    width_factor = style.width_scale * rng.uniform(1 - 0.10*intensity, 1 + 0.10*intensity)
    glyph = glyph.resize((max(1, round(glyph.width * width_factor)), glyph.height), Image.Resampling.LANCZOS)
    angle = style.rotation + rng.uniform(-3.2, 3.2) * intensity
    glyph = glyph.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True)
    if intensity:
        mask = np.asarray(glyph, dtype=np.uint8)
        if medium == "pen" and rng.random() < 0.35 * intensity:
            kernel = np.ones((2, 2), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)
        mask = (
            _apply_pencil_texture(mask, rng, intensity)
            if medium == "pencil" else _apply_ballpoint_texture(mask, rng, intensity)
        )
        blur = rng.uniform(0.08, 0.42) if medium == "pencil" else rng.uniform(0.0, 0.32)
        glyph = Image.fromarray(mask, mode="L").filter(ImageFilter.GaussianBlur(blur))
    return glyph.resize(
        (max(1, round(glyph.width / scale)), max(1, round(glyph.height / scale))),
        Image.Resampling.LANCZOS,
    )


def render_handwritten_text(text: str, width: int, height: int, rng: random.Random,
                            style: HandwritingStyle | None = None,
                            intensity: float = 0.65, medium: str = "pencil",
                            font_family: str = DEFAULT_FONT_FAMILY) -> Image.Image:
    font_file = font_path(font_family)
    if not font_file.is_file():
        raise FileNotFoundError(f"Fonte manuscrita não encontrada: {font_file}")
    if medium not in {"pen", "pencil"}:
        raise ValueError(f"Material de escrita desconhecido: {medium}")
    width, height = max(1, int(width)), max(1, int(height))
    intensity = min(1.0, max(0.0, intensity))
    style = style or random_handwriting_style(rng, intensity, medium=medium)
    glyphs = [_render_glyph(character, height, style, rng, intensity, medium=medium, font_family=font_family) for character in text]
    gap = max(0, round(height * 0.025 * style.spacing_scale))
    content_width = sum(glyph.width for glyph in glyphs) + gap * max(0, len(glyphs)-1)
    mask = Image.new("L", (max(width, content_width + 4), height), 0)
    cursor = 2
    for glyph in glyphs:
        baseline = round(style.baseline_bias * height + rng.uniform(-0.045, 0.045) * height * intensity)
        y = (height - glyph.height) // 2 + baseline


        mask.paste(glyph, (cursor, y))
        cursor += glyph.width + gap
    crop = mask.getbbox()
    if crop is not None:
        content = mask.crop((0, 0, min(mask.width, max(cursor, crop[2]+2)), height))
        if content.width > width:
            content = content.resize((width, height), Image.Resampling.LANCZOS)
        mask = Image.new("L", (width, height), 0)
        if len(text) <= 1:
            # Single characters are composite digit-box cells or checkbox marks
            # (one glyph per box): centring is correct there, like a person filling
            # one printed cell.
            x0 = max(0, (width - content.width) // 2)
        else:
            # Multi-character fields: a person starts writing near the left edge of
            # the line, not centred in the box (baseline writing audit showed
            # left_margin_h 1.56-2.59 -- i.e. text floating in the middle of wide
            # fields -- vs. the typed-text/real-form norm of a small left margin).
            left_pad = max(round(HANDWRITING_LEFT_PAD_HEIGHT_FRAC * height), HANDWRITING_MIN_LEFT_PAD_PX)
            x0 = min(left_pad, max(0, width - content.width))
        mask.paste(content, (x0, 0))
    rgba = Image.new("RGBA", (width, height), (*style.ink, 0))
    rgba.putalpha(mask)
    return rgba


def render_handwritten_numeric(text: str, width: int, height: int, rng: random.Random,
                               style: HandwritingStyle | None = None,
                               intensity: float = 0.65, font_family: str = DEFAULT_FONT_FAMILY) -> Image.Image:
    if not is_numeric_text(text):
        raise ValueError("O renderizador manuscrito aceita somente números e separadores")
    return render_handwritten_text(
        text, width, height, rng, style=style, intensity=intensity, medium="pen", font_family=font_family,
    )


def render_typed_text(text: str, width: int, height: int, font_family: str = DEFAULT_FONT_FAMILY) -> Image.Image:
    """GUI font-preview renderer. Kept in step with `insert_printed`'s placement rules
    (size from the digit-fill target, left padding proportional to height) so the
    preview panel shows what document generation will actually produce."""
    width, height = max(1, int(width)), max(1, int(height))
    font_path_str = str(font_path(font_family))
    digit_ratio = _typed_digit_height_ratio(font_path_str)
    font_size = max(8, round((TYPED_TARGET_DIGIT_HEIGHT_FRAC * height) / digit_ratio))
    size_cap = TYPED_MAX_FONT_SIZE if len(text) <= 1 else TYPED_GENERIC_MAX_SIZE
    font_size = min(font_size, round(height * 1.2), round(size_cap))  # same realistic cap as insert_printed
    font = ImageFont.truetype(font_path_str, font_size)
    left_pad = max(TYPED_MIN_LEFT_PAD_PT, height * TYPED_LEFT_PAD_HEIGHT_FRAC)
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    draw.text((max(2, round(left_pad)), height // 2), text, font=font, fill=255, anchor="lm")
    rgba = Image.new("RGBA", (width, height), (13, 13, 13, 0))
    rgba.putalpha(mask)
    return rgba


def _fit_mark_image(image: Image.Image, width: int, height: int, variant: str) -> Image.Image:
    canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    alpha_bbox = image.getchannel("A").getbbox()
    if alpha_bbox is not None:
        image = image.crop(alpha_bbox)
    if image.width <= 0 or image.height <= 0:
        return canvas
    if variant in {"middle_stroke", "upper_stroke"}:
        target_width, target_height = max(1, round(width * 0.90)), max(1, round(height * 0.42))
    elif variant in {"x", "x_combo"}:
        target_width, target_height = max(1, round(width * 0.78)), max(1, round(height * 0.78))
    elif variant in {"slash", "backslash"}:
        # a single diagonal pen stroke, sized close to what a person actually draws
        # inside a checkbox (not stretched to fill the whole box).
        target_width, target_height = max(1, round(width * 0.58)), max(1, round(height * 0.85))
    else:
        target_width, target_height = max(1, round(width * 0.42)), max(1, round(height * 0.92))
    scale = min(target_width / image.width, target_height / image.height)
    resized = image.resize(
        (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
        Image.Resampling.LANCZOS,
    )
    x = (width - resized.width) // 2
    if variant == "upper_stroke":
        y = max(0, round(height * 0.18) - resized.height // 2)
    else:
        y = (height - resized.height) // 2
    canvas.alpha_composite(resized, (x, min(max(0, y), max(0, height - resized.height))))
    return canvas


def _split_into_word_chunks(text: str, max_chunk_chars: int = 12) -> list[str]:
    """Split text into chunks on whitespace boundaries only, never through a word.

    A single word longer than max_chunk_chars is kept whole (the model must write
    it as one chunk) rather than being cut mid-word.
    """
    words = text.split(" ")
    chunks: list[str] = []
    current = ""
    for word in words:
        if not word:
            continue
        candidate = f"{current} {word}".strip() if current else word
        if current and len(candidate) > max_chunk_chars:
            chunks.append(current)
            current = word
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks or [text]


class _SampleRecorder:
    """Model stand-in that keeps each sample request and returns a short stroke. A request that
    failed in the batch fails again, and a text that the model check refuses fails."""

    def __init__(self, results=None, check=None):
        self.requests, self.results, self.check = [], results or {}, check

    def sample(self, text, writer, bias, seed):
        request = (text, writer, bias, seed)
        if isinstance(self.results.get(request), Exception):
            raise ValueError(str(self.results[request]))
        if self.check is not None:
            self.check(text, writer, bias)
        self.requests.append(request)
        return np.array([[0, 0, 0], [4, 2, 0], [8, 0, 1]], dtype=np.float32)


class _ScratchPage:
    """Page stand-in for the recording pass. It drops the images and sends the other calls to the page."""

    def __init__(self, page):
        self._page = page

    def insert_image(self, *args, **kwargs):
        return 0

    def __getattr__(self, name):
        return getattr(self._page, name)


class _FastImagePage:
    """Page that inserts many images fast. PyMuPDF insert_image scans all page resources on
    each call to find a free fzImg<N> name. This page scans once and gives the same names.
    It calls wrap_contents once: each image block has its own q/Q, so the stream stays
    balanced. It uses the private page._insert_image; FastImagePageTests checks it."""

    def __init__(self, page):
        self._page = page
        doc = page.parent
        used = [i[7] for i in doc.get_page_images(page.number)]
        used += [i[1] for i in doc.get_page_xobjects(page.number)]
        used += [i[4] for i in doc.get_page_fonts(page.number)]
        self._used_names = set(used)
        self._next_index = 0
        self._wrapped = False

    def _next_name(self):
        while True:
            name = f"fzImg{self._next_index}"
            self._next_index += 1
            if name not in self._used_names:
                self._used_names.add(name)
                return name

    def insert_image(self, rect, *, stream, overlay=True, keep_proportion=False):
        page = self._page
        doc = page.parent
        clip = fitz.Rect(rect)
        if clip.is_empty or clip.is_infinite:
            raise ValueError("rect must be finite and not empty")
        clip = clip * ~page.transformation_matrix
        if overlay and not self._wrapped:
            page.wrap_contents()
            self._wrapped = True
        digests = doc.InsertedImages
        xref, digests = page._insert_image(
            filename=None, pixmap=None, stream=stream, imask=None,
            clip=clip, overlay=overlay, oc=0, xref=0, rotate=0,
            keep_proportion=keep_proportion, width=0, height=0, alpha=-1,
            _imgname=self._next_name(), digests=digests,
        )
        if digests is not None:
            doc.InsertedImages = digests
        return xref

    def __getattr__(self, name):
        return getattr(self._page, name)


class _SampleReplay:
    """Model wrapper that returns the batch result of a request. A request that is not in the
    batch calls sample() of the model."""

    def __init__(self, model, results):
        self.model, self.results = model, results
        self.hits = self.misses = 0

    def sample(self, text, writer, bias, seed):
        result = self.results.get((text, writer, bias, seed))
        if result is None:
            self.misses += 1
            return self.model.sample(text, writer, bias, seed)
        self.hits += 1
        if isinstance(result, Exception):
            raise ValueError(str(result))
        return result.copy()


def render_neural_text(text: str, width: int, height: int, pen: NeuralPen,
                       model, rng: random.Random) -> tuple[Image.Image, dict]:
    rendered = prepare_text(text, pen.text_policy)
    chunks = [rendered]
    if pen.model == "sjvasquez" and len(rendered) > 12:
        chunks = _split_into_word_chunks(rendered, max_chunk_chars=12)
    canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    margin = max(1, round(height * 0.08))
    padding = pen.width / 2 + 2
    # Height-driven natural scale per chunk: how big the model would draw this
    # chunk if it had unlimited horizontal room, scaled only so its ink fits the
    # field HEIGHT (never stretched/squashed to fill a pre-assigned width share).
    chunk_points = []
    chunk_scales = []
    chunk_ranges = []
    seeds = []
    total_points = 0
    for chunk in chunks:
        last_error = None
        points = None
        for _ in range(4):
            seed = rng.getrandbits(32)
            try:
                points = model.sample(chunk, pen.writer or 0, pen.bias, seed)
                break
            except ValueError as exc:
                last_error = exc
        if points is None:
            raise ValueError(f"O modelo {pen.model} não conseguiu escrever o valor após novas tentativas") from last_error
        points = trim_trailing_stray_mark(points, pen.width)
        visible = points[:, :2]
        low, high = visible.min(axis=0), visible.max(axis=0)
        range_xy = np.maximum(high - low, 1)
        natural_scale = max((height - 2 * padding) / range_xy[1], 1e-6)
        chunk_points.append((points, seed))
        chunk_scales.append(natural_scale)
        chunk_ranges.append(range_xy)
        seeds.append(seed)
        total_points += len(points)
    natural_widths = [scale * range_xy[0] + 2 * padding for scale, range_xy in zip(chunk_scales, chunk_ranges)]
    word_gap = max(1, round(height * 0.28))
    natural_total = sum(natural_widths) + word_gap * max(0, len(natural_widths) - 1)
    available = max(4, width - margin)
    overall_scale = float(min(1.0, available / natural_total)) if natural_total > 0 else 1.0
    cursor = margin
    for (points, seed), natural_width, range_xy in zip(chunk_points, natural_widths, chunk_ranges):
        final_width = max(4, round(natural_width * overall_scale))
        try:
            image = render_strokes(points, final_width, height, pen, seed)
        except ValueError:
            # The natural box was narrower than the pen padding allows; fall back
            # to a slightly wider box rather than losing the whole chunk.
            final_width = max(final_width, round(2 * padding) + 4)
            image = render_strokes(points, final_width, height, pen, seed)
        canvas.alpha_composite(image, (round(cursor), 0))
        cursor += final_width + word_gap * overall_scale
    return canvas, {"backend": pen.model, "seeds": seeds, "stroke_points": total_points, "chunk_scale": overall_scale}


# Candidate "slash_set": draw the mark as the "/" and/or "\" glyph of a handwriting font
# (the pen stroke used to tick a checkbox), instead of the legacy "I"/"x" letter stroke.
SLASH_VARIANTS = ("slash", "backslash", "x_combo")
SLASH_CHARACTERS = {"slash": ("/",), "backslash": ("\\",), "x_combo": ("/", "\\")}
# Candidate "legacy": the original behaviour (kept so both can be rendered from the same
# seed and compared; see scripts/compare_checkbox_marks.py).
LEGACY_VARIANTS = ("middle_stroke", "upper_stroke", "vertical_stroke", "x")

# Default candidate per backend, chosen from the contact sheets and mark geometry of
# scripts/compare_checkbox_marks.py (legacy vs slash_set, same seeds, DO and DNV boxes).
DEFAULT_MARK_CANDIDATE = {
    "sjvasquez": "slash_set", "wpvnteam": "slash_set",
    "font": "slash_set",
}


def _hex_to_rgb(color: str) -> tuple[int, int, int]:
    return tuple(int(color[index:index + 2], 16) for index in (1, 3, 5))


def _mark_glyph_image(characters: tuple[str, ...], width: int, height: int, rng: random.Random,
                      style: HandwritingStyle, intensity: float, font_family: str) -> Image.Image:
    canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    for character in characters:
        glyph = render_handwritten_text(
            character, width, height, rng, style=style, intensity=intensity, medium="pen", font_family=font_family,
        )
        canvas.alpha_composite(glyph)
    return canvas


def render_checkbox_mark(width: int, height: int, rng: random.Random,
                         style: HandwritingStyle | None = None,
                         intensity: float = 0.65,
                         neural_pen: NeuralPen | None = None,
                         handwriting_model=None,
                         render_rng: random.Random | None = None,
                         font_family: str = DEFAULT_FONT_FAMILY,
                         mark_candidate: str | None = None) -> tuple[Image.Image, dict]:
    """Render one checked-box mark.

    `mark_candidate` selects how the mark is drawn, so the old and new behaviour can be
    compared from the same seed:
      - "legacy": the original "I"/"x" letter stroke (neural pen or font handwriting).
      - "slash_set": the "/" and/or "\\" pen-stroke glyph of a handwriting font (for
        neural presets, tinted to the pen's own colour/width, since the neural models'
        alphabet has no "/" or "\\" glyph to sample).
      - None (default): uses DEFAULT_MARK_CANDIDATE for the active backend.
    """
    width, height = max(1, int(width)), max(1, int(height))
    backend_key = neural_pen.model if neural_pen is not None else "font"
    candidate = mark_candidate or DEFAULT_MARK_CANDIDATE.get(backend_key, "slash_set")
    rotation = rng.uniform(-8, 8)

    if candidate == "legacy":
        variant = rng.choice(LEGACY_VARIANTS)
        character = "x" if variant == "x" else "I"
        if variant in {"middle_stroke", "upper_stroke"}:
            rotation += rng.choice((-90, 90))
        if neural_pen is not None:
            if handwriting_model is None:
                handwriting_model = load_model(neural_pen.model)
            sample_rng = render_rng or rng
            rendered = prepare_text(character, neural_pen.text_policy)
            image, neural_metadata = render_neural_text(rendered, width, height, neural_pen, handwriting_model, sample_rng)
            metadata = {
                **neural_metadata, "backend": neural_pen.model, "mark_variant": variant,
                "requested_value": character, "mark_candidate": candidate,
            }
        else:
            image = render_handwritten_text(
                character, width, height, rng, style=style, intensity=intensity, medium="pen", font_family=font_family,
            )
            metadata = {"backend": "font", "mark_variant": variant, "requested_value": character, "mark_candidate": candidate}
        image = image.rotate(rotation, resample=Image.Resampling.BICUBIC, expand=True)
        return _fit_mark_image(image, width, height, variant), {"rendering": metadata}

    # "slash_set"
    variant = rng.choice(SLASH_VARIANTS)
    characters = SLASH_CHARACTERS[variant]
    if neural_pen is not None:
        # The neural handwriting models' alphabet has no "/" or "\" glyph (see
        # src/neural_handwriting.ALPHABET: it only has letters, digits and a few
        # punctuation marks), so the neural pen itself cannot write these characters.
        # Instead we draw them with the Homemade Apple handwriting font - whose "/" and
        # "\" already look like a natural pen tick - tinted to the neural pen's own ink
        # colour, and thickened to roughly the pen's stroke width, so the mark reads as
        # the same hand as the rest of the field (this also targets the baseline finding
        # that wpvnteam marks were unusually thin, since stroke thickness no longer
        # depends on that model's own synthesis).
        mark_style = HandwritingStyle(
            ink=_hex_to_rgb(neural_pen.color), rotation=0.0, width_scale=1.0, size_scale=1.0,
            spacing_scale=1.0, stroke_bias=1 if neural_pen.width >= 2.2 else 0, baseline_bias=0.0,
        )
        image = _mark_glyph_image(characters, width, height, rng, mark_style, 0.0, "homemade_apple")
        metadata = {"backend": neural_pen.model, "mark_variant": variant, "requested_value": "".join(characters), "mark_candidate": candidate}
    else:
        mark_style = style or random_handwriting_style(rng, intensity, medium="pen")
        image = _mark_glyph_image(characters, width, height, rng, mark_style, intensity, font_family)
        metadata = {"backend": "font", "mark_variant": variant, "requested_value": "".join(characters), "mark_candidate": candidate}
    image = image.rotate(rotation, resample=Image.Resampling.BICUBIC, expand=True)
    return _fit_mark_image(image, width, height, variant), {"rendering": metadata}


def _shared(field: dict, rng: random.Random, context: dict, default_key: str | None,
            factory: Callable[[random.Random], Any], accepts: Callable[[Any], bool]):
    name = validation_rules(field).get("semantic_key") or default_key
    if not name:
        return factory(rng)
    record, document = _record(context, rng), context.get("document")
    target = context.get("links", {}).get(name)
    if target is None:
        defaults = DEFAULT_LINKS.get(name, {})
        target = defaults.get(document, defaults.get(None))
        target = target if target and _allowed(target, document) else None
    if target and accepts(value := record.value(target)):
        return value
    return record.link(f'{field.get("data_type", "text")}:{name}', factory)


def random_date(field: dict, rng: random.Random, context: dict | None = None) -> date:
    return _shared(field, rng, {} if context is None else context, None,
                   lambda source: date(1980, 1, 1) + timedelta(days=source.randrange(0, 46 * 365)),
                   lambda value: isinstance(value, date))


def random_sex(field: dict, rng: random.Random, context: dict | None = None) -> str:
    return _shared(field, rng, {} if context is None else context, "sexo", lambda source: source.choice(("M", "F")),
                   lambda value: value in SEX_VALUES)


def random_dnv(field: dict, rng: random.Random, context: dict | None = None) -> str:
    prefix = str(validation_rules(field).get("prefix") or DNV_TEST_PREFIX)
    return _shared(field, rng, {} if context is None else context, "dnv",
                   lambda source: prefix + f"{source.randrange(10 ** 8):08d}" + str(source.randrange(10)),
                   lambda value: isinstance(value, str) and len(value) == 11 and value.startswith(prefix))


def typed_value(field: dict, rng: random.Random, context: dict | None = None) -> tuple[str, dict]:
    context = {} if context is None else context
    data_type = field.get("data_type", "text")
    validation = validation_rules(field)
    if data_type == "date":
        generated = random_date(field, rng, context)
        return format_date(generated, validation.get("format")), {"date": generated.isoformat()}
    if data_type == "sex":
        code = random_sex(field, rng, context)
        return sex_text(code, validation.get("format")), {"sex": code}
    if data_type == "dnv_number":
        digits = random_dnv(field, rng, context)
        return format_dnv(digits, validation.get("format")), {"identifier": dnv_metadata(digits)}
    return random_value(field, rng, context), {}


def _composite(field: dict) -> bool:
    data_type, fmt = field.get("data_type", "text"), validation_rules(field).get("format")
    return bool(field.get("elements")) and (data_type in {"digit_sequence", "text_sequence"} or (data_type == "date" and fmt == SEPARATED_DATE)
                                            or (data_type == "dnv_number" and fmt == "caixas"))


def _ordered(field: dict) -> list[dict]:
    return sorted(field.get("elements") or [], key=lambda item: item.get("index", 10_000))


def element_values(field: dict, rng: random.Random, context: dict | None = None) -> tuple[str, list[tuple[dict, str]], dict] | None:
    context = {} if context is None else context
    elements = field.get("elements") or []
    data_type = field.get("data_type", "text")
    if not _composite(field):
        return None
    ordered = _ordered(field)
    if data_type in {"digit_sequence", "text_sequence"}:
        sequence = sequence_value(field, ordered, rng, context)
        return sequence, list(zip(ordered, sequence)), {}
    if data_type == "date":
        generated = random_date(field, rng, context)
        parts = [(element, generated.strftime(DATE_PARTS[element["part"]][1])) for element in elements if element.get("part") in DATE_PARTS]
        return format_date(generated, SEPARATED_DATE), parts, {"date": generated.isoformat()}
    digits = random_dnv(field, rng, context)
    return digits, list(zip(ordered, digits)), {"identifier": dnv_metadata(digits)}


def random_value(field: dict, rng: random.Random, context: dict | None = None) -> str:
    value = _random_value(field, rng, {} if context is None else context)
    limit = validation_rules(field).get("max_length")
    return value[:limit] if isinstance(limit, int) and field.get("data_type", "text") != "digit_sequence" else value


def _random_value(field: dict, rng: random.Random, context: dict) -> str:
    data_type = field.get("data_type", "text")
    validation = validation_rules(field)
    allowed = validation.get("allowed_values") or []
    label = _normalized_text(" ".join(str(field.get(key, "")) for key in ("field_id", "label", "description", "notes")))
    if data_type == "digit_sequence":
        return _random_digit_sequence(field, rng, context)
    if data_type in {"date", "sex", "dnv_number"}:
        return typed_value(field, rng, context)[0]
    profile = _location_profile(field, rng, context)
    if profile:
        if data_type == "uf" or " uf" in f" {label}" or label.endswith("uf"):
            return profile["uf"]
        if "cep" in label:
            return profile["cep"]
        if "bairro" in label or "distrito" in label:
            return profile["neighborhood"]
        if "municipio" in label or "cidade" in label:
            return profile["city"]
        if "complemento" in label:
            return rng.choice(COMPLEMENTS)
        if "endereco" in label or "logradouro" in label:
            return profile["street"]
    if allowed:
        return str(rng.choice(allowed))
    if validation.get("mask"):
        return masked_value(str(validation["mask"]), rng)
    if data_type == "municipality":
        return profile["city"] if profile else rng.choice(CITIES)
    if data_type == "address":
        return profile["street"] if profile else f"{rng.choice(STREETS)}, {rng.randrange(1, 1500)}"
    if data_type == "occupation":
        return rng.choice(OCCUPATIONS)
    if data_type == "document":
        return document_number(str(validation.get("document_kind", "cpf")), rng)
    if data_type == "signature":
        return f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"
    if data_type == "uf":
        return rng.choice(UF_VALUES)
    if data_type == "integer":
        return str(_bounded_integer(field, rng, context))
    if data_type == "time":
        return f"{rng.randrange(0, 24):02d}:{rng.randrange(0, 60):02d}"
    if data_type == "person_name" or "nome" in label:
        return f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)} {rng.choice(LAST_NAMES)}"
    if data_type == "date" or "data" in label:
        generated = date(1980, 1, 1) + timedelta(days=rng.randrange(0, 46 * 365))
        return format_date(generated, validation.get("format") if data_type == "date" else None)
    if data_type == "time" or "hora" in label:
        return f"{rng.randrange(0, 24):02d}:{rng.randrange(0, 60):02d}"
    if data_type == "cep" or "cep" in label:
        digits = f"{rng.randrange(0, 100_000_000):08d}"
        return f"{digits[:5]}-{digits[5:]}"
    if data_type == "cns" or "cartão sus" in label or "cartao sus" in label:
        return "".join(str(rng.randrange(10)) for _ in range(15))
    if data_type == "integer" or re.search(r"(?<![a-z0-9])(nº|numero|número|idade|quantidade)(?![a-z0-9])", label):
        return str(_bounded_integer(field, rng, context))
    if data_type == "decimal" or "peso" in label:
        if data_type == "decimal" and ("min_value" in validation or "max_value" in validation or "decimals" in validation):
            minimum = float(validation.get("min_value", 0))
            maximum = float(validation.get("max_value", max(minimum, 100)))
            return f"{rng.uniform(minimum, maximum):.{int(validation.get('decimals', 2))}f}".replace(".", ",")
        return f"{rng.uniform(2.0, 4.5):.3f}".replace(".", ",")
    if "município" in label or "municipio" in label or "cidade" in label:
        return rng.choice(CITIES)
    if "endereço" in label or "endereco" in label or "logradouro" in label:
        return f"{rng.choice(STREETS)}, {rng.randrange(1, 1500)}"
    if "sexo" in label:
        return rng.choice(("M", "F"))
    max_length = int(validation.get("max_length", 24))
    value = rng.choice((rng.choice(CITIES), rng.choice(STREETS), "Não informado", "Hospital Municipal"))
    return _ensure_matches_validation(value[:max_length], field, rng, context)


def sequence_value(field: dict, ordered: list[dict], rng: random.Random, context: dict | None = None) -> str:
    if field.get("data_type") == "digit_sequence":
        sequence = random_value({**field, "elements": ordered}, rng, context)
    else:
        sequence = "".join(rng.choice(string.ascii_uppercase) for _ in ordered)
    if len(sequence) < len(ordered):
        sequence = sequence + "".join(rng.choice(string.digits) for _ in range(len(ordered) - len(sequence)))
    return sequence[:len(ordered)]


def _bound_values(field: dict, binding: Binding, value) -> tuple[str, list[tuple[dict, str]] | None, dict]:
    data_type = field.get("data_type", "text")
    typed = {}
    if data_type == "date" and isinstance(value, date) and not binding.part:
        typed = {"date": value.isoformat()}
    elif data_type == "sex" and value in SEX_VALUES:
        typed = {"sex": value}
    elif data_type == "dnv_number" and value_kind(binding, value) == "dnv":
        typed = {"identifier": dnv_metadata(value)}
    if isinstance(value, int) and isinstance(field.get("measurement"), dict):
        measurement = field["measurement"]
        typed["measurement"] = {"value": round(value * measurement["scale"], 1), "unit": measurement["unit"]}
    parts = present_elements(field, binding, value, _ordered(field)) if _composite(field) else None
    return present_text(field, binding, value), parts, typed


def _semantic_entry(binding: Binding, snapshot: dict) -> dict:
    entry = {"key": binding.key, "value": snapshot[binding.key], "source": binding.source}
    entry.update({name: getattr(binding, name) for name in ("part", "format") if getattr(binding, name)})
    return {"semantic": entry}


def field_texts(field: dict, rng: random.Random, context: dict | None = None) -> list[str]:
    if field.get("data_type") == "checkbox" and field.get("elements"):
        return []
    context = {} if context is None else context
    key = field.get("semantic")
    if isinstance(key, str) and key.startswith("custom."):
        definition = (context.get("semantic_definitions") or {}).get(key)
        if definition is None:
            raise ValueError(f"Definição semântica personalizada ausente: {key}")
        value = generate_custom_value(definition, rng)
        binding = Binding(key, source="custom")
        text, parts, _ = _bound_values(field, binding, value)
        return [character for _, character in parts] if parts is not None else [text]
    record = _record(context, rng)
    binding = semantic_plan([field], context.get("document")).apply(record).get(str(field.get("field_id", "")))
    if binding is not None:
        if (value := record.value(binding.key)) is None:
            return []
        text, parts, _ = _bound_values(field, binding, value)
        return [character for _, character in parts] if parts is not None else [text]
    composite = element_values(field, rng, context)
    if composite:
        return [text for _, text in composite[1]]
    return [typed_value(field, rng, context)[0]]


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", value).strip("_")
    return cleaned or "synthetic"


def _generation_options(handwriting, handwriting_intensity, text_handwriting, neural_pen,
                        numeric_neural_pen, text_neural_pen, realistic_filters, filter_preset, style_preset_uid,
                        font_family: str = DEFAULT_FONT_FAMILY,
                        content_pens: dict | None = None, content_fonts: dict | None = None,
                        content_intensity: dict | None = None, content_handwriting: dict | None = None) -> dict:
    options = {"handwriting": handwriting, "handwriting_intensity": handwriting_intensity,
            "text_handwriting": text_handwriting, "neural_pen": asdict(neural_pen) if neural_pen else None,
            "numeric_neural_pen": asdict(numeric_neural_pen) if numeric_neural_pen else None,
            "text_neural_pen": asdict(text_neural_pen) if text_neural_pen else None,
            "realistic_filters": realistic_filters, "filter_preset": filter_preset, "style_preset_uid": style_preset_uid,
            "font_family": font_family}
    if content_pens or content_fonts:
        options["content_pens"] = {k: asdict(v) if v else None for k, v in (content_pens or {}).items()}
        options["content_fonts"] = dict(content_fonts or {})
        options["content_intensity"] = dict(content_intensity or {})
        options["content_handwriting"] = dict(content_handwriting or {})
    return options


def generate_document(template_path: Path, field_map: dict, selected_ids: set[str],
                      output_dir: Path, document_id: str, rng: random.Random,
                      handwriting: bool = True, handwriting_intensity: float = 0.65,
                      text_handwriting: bool = True, neural_pen: NeuralPen | None = None,
                      numeric_neural_pen: NeuralPen | None = None,
                      text_neural_pen: NeuralPen | None = None,
                      handwriting_model=None, realistic_filters: bool = False,
                      filter_preset: str = "medium",
                      font_family: str = DEFAULT_FONT_FAMILY,
                      style_preset_uid: str | None = None,
                      generation_metadata: dict | None = None,
                      record_factory: RecordFactory | None = None,
                      content_pens: dict[str, NeuralPen | None] | None = None,
                      content_fonts: dict[str, str] | None = None,
                      content_intensity: dict[str, float] | None = None,
                      content_handwriting: dict[str, bool] | None = None) -> SyntheticDocument:
    field_map = apply_shared_rules(field_map)
    reproduction = dict(generation_metadata or {})
    if not reproduction:
        reproduction = {"random_state": rng.getstate(), "mapping_snapshot": field_map,
                        "template_sha256": hashlib.sha256(template_path.read_bytes()).hexdigest(),
                        "selected_ids": sorted(selected_ids),
                        "options": _generation_options(handwriting, handwriting_intensity, text_handwriting, neural_pen,
                                                       numeric_neural_pen, text_neural_pen, realistic_filters, filter_preset,
                                                       style_preset_uid, font_family, content_pens, content_fonts, content_intensity, content_handwriting)}
    output_dir.mkdir(parents=True, exist_ok=True)
    document = fitz.open(template_path)
    if len(document) == 0:
        document.close()
        raise ValueError("O template não contém páginas")
    if len(document) > 1:
        document.select([0])
    page = _FastImagePage(document[0])
    dpi = float(field_map.get("dpi", 300))
    point_scale = 72.0 / dpi
    values: dict[str, str] = {}
    entries: list[dict] = []
    split_neural_pens = numeric_neural_pen is not None or text_neural_pen is not None
    if split_neural_pens:
        numeric_neural_pen = numeric_neural_pen or neural_pen
        text_neural_pen = text_neural_pen or neural_pen
    else:
        numeric_neural_pen = neural_pen
        text_neural_pen = neural_pen
    neural_pens = [pen for pen in (numeric_neural_pen, text_neural_pen, *(content_pens or {}).values()) if pen is not None]
    render_rng = random.Random(rng.getrandbits(64)) if neural_pens else None
    handwriting_models = handwriting_model if isinstance(handwriting_model, dict) else {}
    if handwriting_model is not None and not isinstance(handwriting_model, dict) and neural_pen is not None:
        handwriting_models = {neural_pen.model: handwriting_model}

    def resolve_pen(pen: NeuralPen | None) -> NeuralPen | None:
        if pen is None:
            return None
        if pen.writer is None:
            count = 13 if pen.model == "sjvasquez" else 512
            pen = replace(pen, writer=render_rng.randrange(count))
        if pen.model not in handwriting_models:
            handwriting_models[pen.model] = load_model(pen.model)
        return pen

    if split_neural_pens:
        numeric_neural_pen = resolve_pen(numeric_neural_pen)
        text_neural_pen = resolve_pen(text_neural_pen)
    else:
        neural_pen = resolve_pen(neural_pen)
        numeric_neural_pen = neural_pen
        text_neural_pen = neural_pen
    content_pens = content_pens or {}
    content_fonts = content_fonts or {}
    content_intensity = content_intensity or {}
    content_handwriting = content_handwriting or {}
    content_pens = {content_class: resolve_pen(pen) for content_class, pen in content_pens.items()}
    handwriting_style = random_handwriting_style(rng, handwriting_intensity, medium="pen")
    pencil_style = random_handwriting_style(rng, handwriting_intensity, medium="pencil")
    semantic_seed = rng.getrandbits(64)
    generation_context = {"record": None, "document": document_context(field_map.get("template_id") or infer_document_type_key(template_path))}
    selected = [field for field in field_map.get("fields", []) if str(field["field_id"]) in selected_ids]
    unsupported = next((field for field in selected if field.get("unsupported_reason")), None)
    if unsupported:
        raise ValueError(f"{unsupported.get('label') or unsupported['field_id']}: {unsupported['unsupported_reason']}")
    custom_definitions = field_map.get("semantic_definitions") or {}
    custom_fields = [field for field in selected if isinstance(field.get("semantic"), str)
                     and field["semantic"].startswith("custom.")]
    custom_values = {}
    for field in custom_fields:
        key = field["semantic"]
        definition = custom_definitions.get(key)
        if definition is None:
            raise ValueError(f"Definição semântica personalizada ausente: {key}")
        validate_custom_definition(*(definition[name] for name in ("key", "label", "description", "data_type", "generator")))
        if key not in custom_values:
            custom_values[key] = generate_custom_value(definition, random.Random(f"{semantic_seed}:custom:{key}"))
    plan = semantic_plan([field for field in selected if field not in custom_fields], generation_context["document"])
    linked = plan.bindings
    generation_context["links"] = {key: linked[str(field["field_id"])].key for field in selected
                                   if str(field["field_id"]) in linked and (key := validation_rules(field).get("semantic_key"))}
    legacy_keys = [key for field in selected if field not in custom_fields and str(field["field_id"]) not in linked
                   for key in _legacy_keys(field, generation_context)]
    if record_factory is None:
        record = SemanticRecord(semantic_seed, generation_context["document"])
        bindings = plan.apply(record, legacy_keys)
    else:
        record = record_factory(semantic_seed, generation_context["document"], plan, legacy_keys)
        bindings = plan.bindings
    generation_context["record"] = record
    bindings.update({str(field["field_id"]): Binding(field["semantic"], source="custom") for field in custom_fields})
    ai_values = getattr(record, "ai_values", None) or {}
    fields_by_id = {str(field["field_id"]): field for field in selected}
    ai_bindings = {field_id: Binding(ai_binding_key(field_id, fields_by_id[field_id].get("data_type", "text")), source="ai")
                  for field_id in ai_values if field_id in fields_by_id}
    bindings.update(ai_bindings)
    ai_store = {binding.key: ai_values[field_id] for field_id, binding in ai_bindings.items()}
    snapshot = record.snapshot()
    snapshot.update({key: value.isoformat() if isinstance(value, (date, time)) else value for key, value in custom_values.items()})
    snapshot.update({binding.key: (value.isoformat() if isinstance(value, (date, time)) else value)
                     for field_id, binding in ai_bindings.items() for value in (ai_values[field_id],)})

    def semantic_value(binding: Binding):
        if binding.source == "custom":
            return custom_values[binding.key]
        if binding.source == "ai":
            return ai_store[binding.key]
        return record.value(binding.key)

    typed_font = is_typed(font_family)
    typed_alias = TYPED_FONT_ALIAS_PREFIX + font_family
    typed_path = str(font_path(font_family))

    def _resolve_font(content_class: str | None) -> tuple[str, bool, str, str]:
        """Font family, typed flag, alias and path for `content_class`, or
        the document's default font_family when the class has no override."""
        family = content_fonts.get(content_class, font_family) if content_class else font_family
        if family == font_family:
            return font_family, typed_font, typed_alias, typed_path
        return family, is_typed(family), TYPED_FONT_ALIAS_PREFIX + family, str(font_path(family))

    def insert_printed(rect: fitz.Rect, text: str, font_size: float | None, fontname: str, color: tuple[float, float, float],
                       fontfile: str | None = None) -> bool:
        """Place typed/printed text in `rect`: font size derived from field height
        (targeting a realistic digit-fill ratio instead of the old hard 11pt cap), a
        left padding proportional to field height but capped as a fraction of field
        width so narrow fields aren't over-padded (single characters -- composite
        digit-box cells, checkbox marks -- stay centred, no padding), vertical
        centring on the glyph's own ink span, and shrink-to-fit by width down to a
        legibility floor (with one further step down to an absolute hard floor for
        values that still don't fit). Returns False, WITHOUT drawing anything, if the
        text cannot be made to fit even at the hard floor -- OCR training data must
        never show text clipped against the field border, so the caller must treat a
        False return as "could not render" and fall back to leaving the field blank
        with a reason, not accept a clipped image."""
        single_char = len(text) == 1
        face = _typed_font_face(fontfile, fontname)
        if font_size:
            size = font_size
        else:
            digit_ratio = _typed_digit_height_ratio(fontfile) if fontfile else 0.62
            size = (TYPED_TARGET_DIGIT_HEIGHT_FRAC * rect.height) / digit_ratio
            size_cap = TYPED_MAX_FONT_SIZE if single_char else TYPED_GENERIC_MAX_SIZE
            size = max(TYPED_MIN_FONT_SIZE, min(size_cap, size))
        if single_char:
            left_pad = 0.0
        else:
            left_pad = max(TYPED_MIN_LEFT_PAD_PT, min(rect.height * TYPED_LEFT_PAD_HEIGHT_FRAC,
                                                       rect.width * TYPED_LEFT_PAD_MAX_WIDTH_FRAC))
        available_width = max(1.0, rect.width - left_pad - TYPED_RIGHT_PAD_PT)
        text_width = face.text_length(text, fontsize=size)
        while text_width > available_width and size > TYPED_SHRINK_FLOOR:
            size = max(TYPED_SHRINK_FLOOR, size * TYPED_SHRINK_STEP)
            text_width = face.text_length(text, fontsize=size)
        if text_width > available_width:
            # Still doesn't fit at the normal legibility floor: try once more, down to
            # the absolute hard floor, rather than ever drawing clipped text.
            while text_width > available_width and size > TYPED_HARD_FLOOR:
                size = max(TYPED_HARD_FLOOR, size * TYPED_SHRINK_STEP)
                text_width = face.text_length(text, fontsize=size)
        if text_width > available_width:
            return False
        ascender, descender = face.ascender, face.descender
        glyph_span = max(1e-6, (ascender - descender)) * size
        top_pad = max(0.0, (rect.height - glyph_span) / 2.0)
        baseline_y = rect.y0 + top_pad + ascender * size
        # Keep the glyph's own ink span inside the field box: ascenders must not rise
        # above the top, and -- critically for "fill in the line" fields whose bottom
        # edge is a printed underline -- descenders must not cross the bottom.
        baseline_ceiling = rect.y1 + descender * size  # descender <= 0, so this is <= rect.y1
        baseline_floor = rect.y0 + ascender * size
        if baseline_ceiling >= baseline_floor:
            baseline_y = min(max(baseline_y, baseline_floor), baseline_ceiling)
        else:
            baseline_y = baseline_floor  # glyph taller than the box; best effort, top-aligned
        if single_char:
            x0 = rect.x0 + max(0.0, (rect.width - text_width) / 2.0)
        else:
            x0 = rect.x0 + left_pad
        kwargs = {"fontfile": fontfile} if fontfile else {}
        page.insert_text((x0, baseline_y), text, fontsize=size, fontname=fontname, color=color, **kwargs)
        return True

    def insert_value(rect_values: list[int], text: str, font_size: float | None = None,
                     pencil: bool = False, force_printed: bool = False,
                     content_class: str | None = None) -> tuple[str, dict]:
        x1, y1, x2, y2 = rect_values
        rect = fitz.Rect(x1 * point_scale, y1 * point_scale, x2 * point_scale, y2 * point_scale)
        if not text.strip():
            return text, {}
        has_class_override = content_class is not None and (content_class in content_pens or content_class in content_fonts)
        if not has_class_override:
            if typed_font:
                fits = insert_printed(rect, text, font_size, typed_alias, (0.05, 0.05, 0.05), fontfile=typed_path)
                if fits:
                    return text, {}
                return "", {"blank": True, "blank_reason": "Valor não coube no campo em nenhum tamanho legível",
                            "requested_value": text}
            active_pen = numeric_neural_pen if is_numeric_text(text) else text_neural_pen
            neural_enabled = handwriting if is_numeric_text(text) else text_handwriting
            local_font_family = font_family
        else:
            active_pen = content_pens.get(content_class)
            neural_enabled = active_pen is not None
            local_font_family, local_typed, local_typed_alias, local_typed_path = _resolve_font(content_class)
            if local_typed and active_pen is None:
                fits = insert_printed(rect, text, font_size, local_typed_alias, (0.05, 0.05, 0.05), fontfile=local_typed_path)
                if fits:
                    return text, {}
                return "", {"blank": True, "blank_reason": "Valor não coube no campo em nenhum tamanho legível",
                            "requested_value": text}
        if (
            not force_printed
            and active_pen is not None
            and neural_enabled
        ):
            rendered = prepare_text(text, active_pen.text_policy)
            model = handwriting_models[active_pen.model]
            try:
                image, rendering = render_neural_text(rendered, x2-x1, y2-y1, active_pen, model, render_rng)
            except ValueError as exc:
                # The neural model failed on this value after retries: fall back to a
                # clean font handwriting render for this FIELD only (not the whole
                # document), using the same ink colour as the neural pen.
                fallback_style = random_handwriting_style(rng, handwriting_intensity, medium="pen")
                fallback_style = replace(fallback_style, ink=tuple(
                    int(active_pen.color[i:i+2], 16) for i in (1, 3, 5)
                ))
                image = render_handwritten_text(
                    rendered, max(1, x2-x1), max(1, y2-y1), rng,
                    style=fallback_style, intensity=handwriting_intensity, medium="pen", font_family=font_family,
                )
                stream = io.BytesIO()
                image.save(stream, format="PNG")
                page.insert_image(rect, stream=stream.getvalue(), overlay=True, keep_proportion=False)
                return rendered, {"requested_value": text, "rendering": {
                    "backend": "font", "fallback": {"from_backend": active_pen.model, "reason": str(exc)},
                }}
            stream = io.BytesIO()
            image.save(stream, format="PNG")
            page.insert_image(rect, stream=stream.getvalue(), overlay=True, keep_proportion=False)
            return rendered, {"requested_value": text, "rendering": rendering}
        if has_class_override:
            # The class override chose a "font" model (no neural pen) and a
            # non-typed font family: render it as handwriting with that
            # family (numeric glyph shapes for numeric-looking values,
            # cursive/print shapes otherwise) UNLESS the override explicitly
            # turned handwriting off for this class (content_handwriting is
            # False), in which case it falls through to the plain printed
            # fallback below, same as a class with no handwriting at all.
            if not force_printed and not local_typed and content_handwriting.get(content_class, True):
                local_intensity = content_intensity.get(content_class, handwriting_intensity)
                if is_numeric_text(text):
                    image = render_handwritten_numeric(
                        text, max(1, x2-x1), max(1, y2-y1), rng,
                        style=handwriting_style, intensity=local_intensity, font_family=local_font_family,
                    )
                else:
                    image = render_handwritten_text(
                        text, max(1, x2-x1), max(1, y2-y1), rng,
                        style=pencil_style, intensity=local_intensity, medium="pencil", font_family=local_font_family,
                    )
                stream = io.BytesIO()
                image.save(stream, format="PNG")
                page.insert_image(rect, stream=stream.getvalue(), overlay=True, keep_proportion=False)
                return text, {}
        elif not force_printed and handwriting and is_numeric_text(text):
            image = render_handwritten_numeric(
                text, max(1, x2-x1), max(1, y2-y1), rng,
                style=handwriting_style, intensity=handwriting_intensity, font_family=font_family,
            )
            stream = io.BytesIO()
            image.save(stream, format="PNG")
            page.insert_image(rect, stream=stream.getvalue(), overlay=True, keep_proportion=False)
            return text, {}
        elif not force_printed and text_handwriting and pencil:
            image = render_handwritten_text(
                text, max(1, x2-x1), max(1, y2-y1), rng,
                style=pencil_style, intensity=handwriting_intensity, medium="pencil", font_family=font_family,
            )
            stream = io.BytesIO()
            image.save(stream, format="PNG")
            page.insert_image(rect, stream=stream.getvalue(), overlay=True, keep_proportion=False)
            return text, {}
        fits = insert_printed(rect, text, font_size, "helv", (0.05, 0.15, 0.55))
        if fits:
            return text, {}
        return "", {"blank": True, "blank_reason": "Valor não coube no campo em nenhum tamanho legível",
                    "requested_value": text}

    def fill_fields():
        for field in field_map.get("fields", []):
            field_id = str(field["field_id"])
            if field_id not in selected_ids:
                continue
            elements = field.get("elements") or []
            blank_probability = float((field.get("validation") or {}).get("blank_probability") or 0)
            if blank_probability > 0 and rng.random() < blank_probability:
                values[field_id] = ""
                entries.append({"field_id": field_id, "label": field.get("label", ""), "value": "", "bbox": field["bbox"], "blank": True})
                continue
            binding = bindings.get(field_id)
            meaning = _semantic_entry(binding, snapshot) if binding else {}
            render_when = field.get("render_when")
            if render_when and snapshot.get(render_when["key"]) not in render_when["values"]:
                values[field_id] = ""
                entries.append({"field_id": field_id, "label": field.get("label", ""), "value": "", "bbox": field["bbox"],
                                "blank": True, "render_condition": render_when})
                continue
            if binding and snapshot[binding.key] is None:
                values[field_id] = ""
                entries.append({"field_id": field_id, "label": field.get("label", ""), "value": "", "bbox": field["bbox"], "blank": True,
                                **meaning})
                continue
            if field.get("data_type") == "checkbox" and elements:
                if binding:
                    selected_elements = checkbox_choice(field, binding, semantic_value(binding))
                else:
                    count = rng.randint(0, 1) if len(elements) == 1 else (1 if field.get("selection_mode", "single") == "single" else rng.randint(1, len(elements)))
                    selected_elements = rng.sample(elements, count)
                element_entries = []
                for element in elements:
                    checked = element in selected_elements
                    mark_metadata = {}
                    if checked:
                        ex1, ey1, ex2, ey2 = element["bbox"]
                        rect = fitz.Rect(*(v * point_scale for v in element["bbox"]))
                        if typed_font and numeric_neural_pen is None:
                            insert_printed(rect, "X", None, typed_alias, (0.05, 0.05, 0.05), fontfile=typed_path)
                            mark_metadata = {"rendering": {"backend": "font", "mark_variant": "x", "requested_value": "X"}}
                        else:
                            image, mark_metadata = render_checkbox_mark(
                                ex2-ex1, ey2-ey1, rng, style=handwriting_style,
                                intensity=handwriting_intensity, neural_pen=numeric_neural_pen,
                                handwriting_model=handwriting_models.get(numeric_neural_pen.model) if numeric_neural_pen else None,
                                render_rng=render_rng, font_family=font_family,
                            )
                            stream = io.BytesIO()
                            image.save(stream, format="PNG")
                            page.insert_image(rect, stream=stream.getvalue(), overlay=True, keep_proportion=False)
                    element_entries.append({**element, "checked": checked, **mark_metadata})
                value = ",".join(str(item.get("value") or item["element_id"]) for item in selected_elements)
                values[field_id] = value
                entries.append({
                    "field_id": field_id, "label": field.get("label", ""), "value": value,
                    "bbox": field["bbox"], "elements": element_entries, **meaning,
                })
                continue
            if binding:
                value, parts, semantic = _bound_values(field, binding, semantic_value(binding))
                composite = None if parts is None else (value, parts, semantic)
            else:
                composite = element_values(field, rng, generation_context)
            if composite:
                requested, parts, semantic = composite
                field_content_class = classify_content(
                    field.get("data_type", "text"), binding.key if binding else None, requested,
                )
                element_entries = []
                for element, character in parts:
                    rendered, metadata = insert_value(
                        element["bbox"], character,
                        pencil=field.get("data_type") == "text_sequence",
                        force_printed=bool(field.get("render_as_printed", False)),
                        content_class=field_content_class,
                    )
                    element_entries.append({**element, "value": rendered, **metadata})
                sequence = "".join(element["value"] for element in element_entries) if field.get("data_type") in {"digit_sequence", "text_sequence"} else requested
                values[field_id] = sequence
                entries.append({
                    "field_id": field_id, "label": field.get("label", ""), "value": sequence,
                    "bbox": field["bbox"], "elements": element_entries, **semantic, **meaning,
                    **({"requested_value": requested} if (numeric_neural_pen or text_neural_pen) is not None else {}),
                })
                continue
            if not binding:
                value, semantic = typed_value(field, rng, generation_context)
            value, metadata = insert_value(
                field["bbox"], value,
                pencil=field.get("data_type", "text") in TEXT_TYPES and not is_numeric_text(value),
                force_printed=bool(field.get("render_as_printed", False)),
                content_class=classify_content(field.get("data_type", "text"), binding.key if binding else None, value),
            )
            values[field_id] = value
            entries.append({"field_id": field_id, "label": field.get("label", ""), "value": value, "bbox": field["bbox"], **semantic, **meaning,
                            **metadata})

    batched = {name for name, model in handwriting_models.items() if callable(getattr(type(model), "sample_many", None))}
    if batched:
        # Fill the fields on a scratch page with a recorder in place of each model, and sample the
        # recorded requests in one batch. The seeds do not depend on the strokes, thus the real pass
        # makes the same requests. A failed request makes a retry with a new seed, thus the recorder
        # fails the known failed requests and the planning pass runs again. A request that is not in
        # the batch calls sample(). An error in the planning pass only causes such misses.
        saved = (rng.getstate(), render_rng.getstate(), page, field_map, generation_context, record, snapshot,
                 values, entries, handwriting_models)
        results = {name: {} for name in batched}
        for _ in range(4):
            recorders = {name: _SampleRecorder(results.get(name), model.check if name in batched else None)
                         for name, model in handwriting_models.items()}
            scratch = fitz.open(template_path)
            field_map, generation_context, record, snapshot = copy.deepcopy(saved[3:7])
            page, values, entries, handwriting_models = _ScratchPage(scratch[0]), {}, [], recorders
            try:
                fill_fields()
            except Exception:
                pass
            finally:
                scratch.close()
            state, render_state, page, field_map, generation_context, record, snapshot, values, entries, handwriting_models = saved
            rng.setstate(state)
            render_rng.setstate(render_state)
            failed = False
            for name in batched:
                requests = [request for request in dict.fromkeys(recorders[name].requests) if request not in results[name]]
                results[name].update(zip(requests, handwriting_models[name].sample_many(requests)))
                failed = failed or any(isinstance(results[name][request], Exception) for request in requests)
            if not failed:
                break
        handwriting_models = {name: _SampleReplay(model, results[name]) if name in batched else model
                              for name, model in handwriting_models.items()}
    fill_fields()

    file_document_id = document_id_with_preset(document_id, style_preset_uid)
    safe_id = _safe_name(file_document_id)
    pdf_path = output_dir / f"{safe_id}.pdf"
    json_path = output_dir / f"{safe_id}.json"
    document.save(pdf_path, garbage=3, deflate=True, deflate_images=True, deflate_fonts=True, use_objstms=1)
    document.close()
    visual_filters = {"enabled": False}
    if realistic_filters:
        visual_filters = apply_realistic_filters_to_pdf(
            pdf_path, dpi=round(dpi), rng=random.Random(rng.getrandbits(64)),
            preset=filter_preset,
        )
    handwriting_metadata = (
        (neural_pen.metadata() if neural_pen is not None else {"model": "font"})
        if not split_neural_pens else {
            "numbers": numeric_neural_pen.metadata() if numeric_neural_pen is not None else {"model": "font"},
            "text": text_neural_pen.metadata() if text_neural_pen is not None else {"model": "font"},
        }
    )
    ground_truth = {
        "document_id": file_document_id,
        "template_id": field_map.get("template_id", template_path.stem),
        "source_template": str(template_path),
        "fields": entries,
        "handwriting": handwriting_metadata,
        "style_preset_uid": style_preset_uid,
        "visual_filters": visual_filters,
        "semantic_version": SEMANTIC_VERSION,
        "semantic_seed": record.seed,
        "semantic_context": record.context,
        "semantic_record": snapshot,
        **({"semantic_provenance": record.provenance} if record.provenance else {}),
        **({"ai_values": serialize_ai_values(ai_values)} if ai_values else {}),
        "generation": reproduction,
    }
    json_path.write_text(json.dumps(ground_truth, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return SyntheticDocument(pdf_path=pdf_path, json_path=json_path, values=values)


def generate_dataset(template_path: Path, mapping_path: Path, selected_ids: set[str],
                     output_dir: Path | None, count: int, seed: int | None = None,
                     handwriting: bool = True,
                     handwriting_intensity: float = 0.65,
                     text_handwriting: bool = True,
                     neural_pen: NeuralPen | None = None,
                     numeric_neural_pen: NeuralPen | None = None,
                     text_neural_pen: NeuralPen | None = None,
                     realistic_filters: bool = False,
                     filter_preset: str = "medium",
                     style_preset_uid: str | None = None,
                     font_family: str = DEFAULT_FONT_FAMILY,
                     progress_callback: Callable[[int, int, SyntheticDocument], None] | None = None,
                     record_factory: Callable[[int, int, str | None, SemanticPlan, list[str]], SemanticRecord] | None = None,
                     content_pens: dict[str, NeuralPen | None] | None = None,
                     content_fonts: dict[str, str] | None = None,
                     content_intensity: dict[str, float] | None = None,
                     content_handwriting: dict[str, bool] | None = None,
                     ) -> list[SyntheticDocument]:
    if count <= 0:
        raise ValueError("A quantidade deve ser maior que zero")
    if not selected_ids:
        raise ValueError("Selecione ao menos um campo")
    if output_dir is None:
        document_type = infer_document_type_key(template_path, mapping_path)
        output_dir = build_dataset_run_dir(document_type, count)
    field_map = json.loads(mapping_path.read_text(encoding="utf-8"))
    field_map = apply_shared_rules(field_map)
    unsupported = next((field for field in field_map.get("fields", [])
                        if str(field["field_id"]) in selected_ids and field.get("unsupported_reason")), None)
    if unsupported:
        raise ValueError(f"{unsupported.get('label') or unsupported['field_id']}: {unsupported['unsupported_reason']}")
    requested_seed = seed
    seed = secrets.randbits(32) if seed is None else seed
    rng = random.Random(seed)
    options = _generation_options(handwriting, handwriting_intensity, text_handwriting, neural_pen,
                                  numeric_neural_pen, text_neural_pen, realistic_filters, filter_preset, style_preset_uid,
                                  font_family, content_pens, content_fonts, content_intensity, content_handwriting)
    output_dir.mkdir(parents=True, exist_ok=True)
    mapping_bytes = (json.dumps(field_map, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    mapping_hash = hashlib.sha256(mapping_bytes).hexdigest()
    reproduction = {"schema_version": 1, "mapping_snapshot": field_map,
                    "mapping_sha256": mapping_hash,
                    "mapping_version": field_map.get("version", field_map.get("mapping_version", mapping_hash)),
                    "template_sha256": hashlib.sha256(template_path.read_bytes()).hexdigest(),
                    "selected_ids": sorted(selected_ids), "options": options}
    split_neural_pens = numeric_neural_pen is not None or text_neural_pen is not None
    if split_neural_pens:
        numeric_neural_pen = numeric_neural_pen or neural_pen
        text_neural_pen = text_neural_pen or neural_pen
    else:
        numeric_neural_pen = neural_pen
        text_neural_pen = neural_pen
    content_pens = content_pens or {}
    content_fonts = content_fonts or {}
    content_intensity = content_intensity or {}
    content_handwriting = content_handwriting or {}
    handwriting_model = {
        model: load_model(model)
        for model in {pen.model for pen in (numeric_neural_pen, text_neural_pen, *content_pens.values()) if pen is not None}
    }
    generated = []
    for index in range(1, count + 1):
        document = generate_document(
            template_path, field_map, selected_ids, output_dir, f"synthetic_{index:04d}", rng,
            handwriting=handwriting, handwriting_intensity=handwriting_intensity,
            text_handwriting=text_handwriting,
            neural_pen=neural_pen,
            numeric_neural_pen=numeric_neural_pen if split_neural_pens else None,
            text_neural_pen=text_neural_pen if split_neural_pens else None,
            handwriting_model=handwriting_model,
            realistic_filters=realistic_filters, filter_preset=filter_preset,
            style_preset_uid=style_preset_uid, font_family=font_family,
            content_pens=content_pens, content_fonts=content_fonts, content_intensity=content_intensity,
            content_handwriting=content_handwriting,
            generation_metadata={"manifest": "dataset.json", "seed": seed, "index": index,
                                 "mapping_sha256": mapping_hash, "template_sha256": reproduction["template_sha256"]},
            record_factory=None if record_factory is None else partial(record_factory, index),
        )
        generated.append(document)
        if progress_callback is not None:
            progress_callback(index, count, document)
    handwriting_metadata = (
        (neural_pen.metadata() if neural_pen is not None else {"model": "font"})
        if not split_neural_pens else {
            "numbers": numeric_neural_pen.metadata() if numeric_neural_pen is not None else {"model": "font"},
            "text": text_neural_pen.metadata() if text_neural_pen is not None else {"model": "font"},
        }
    )
    manifest = {
        "template": str(template_path), "mapping": str(mapping_path),
        "seed": seed, "count": len(generated),
        "requested_seed": requested_seed,
        "reproduction": reproduction,
        "handwriting": {
            **handwriting_metadata,
            "enabled": handwriting, "numbers_enabled": handwriting,
            "text_enabled": text_handwriting,
            "intensity": handwriting_intensity,
            "font_family": font_family,
        },
        "visual_filters": {"enabled": realistic_filters, "preset": filter_preset},
        "style_preset_uid": style_preset_uid,
        "semantic_version": SEMANTIC_VERSION,
        "documents": [{"pdf": item.pdf_path.name, "ground_truth": item.json_path.name} for item in generated],
    }
    (output_dir / "dataset.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return generated
