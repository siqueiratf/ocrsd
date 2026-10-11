from __future__ import annotations

import json
import io
import random
import re
import string
import unicodedata
from dataclasses import dataclass, replace
from datetime import date, timedelta
from pathlib import Path
from typing import Callable

import fitz
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from src.document_types import build_dataset_run_dir, infer_document_type_key
from src.document_filters import apply_realistic_filters_to_pdf
from src.neural_handwriting import NeuralPen, load_model, prepare_text, render_strokes
from src.writing_presets import document_id_with_preset


ASSETS_ROOT = Path(__file__).resolve().parent.parent / "assets"
HANDWRITING_FONT = ASSETS_ROOT / "fonts" / "Caveat-Regular.ttf"
NUMERIC_CHARACTERS = frozenset("0123456789.,:/- ")


FIRST_NAMES = (
    "Ana", "Beatriz", "Camila", "Clara", "Daniel", "Eduardo", "Felipe",
    "Gabriel", "Helena", "Isabela", "João", "Júlia", "Lucas", "Mariana",
    "Miguel", "Rafael", "Sofia", "Valentina",
)
LAST_NAMES = (
    "Almeida", "Barbosa", "Cardoso", "Costa", "Ferreira", "Gomes",
    "Lima", "Martins", "Oliveira", "Pereira", "Ribeiro", "Rodrigues",
    "Santos", "Silva", "Souza",
)
STREETS = ("Rua das Flores", "Avenida Brasil", "Rua São José", "Rua das Acácias", "Avenida Central")
CITIES = ("São Paulo", "Rio de Janeiro", "Belo Horizonte", "Curitiba", "Recife", "Salvador")
UF_VALUES = ("AC", "AL", "AP", "AM", "BA", "CE", "DF", "ES", "GO", "MA", "MT", "MS", "MG", "PA", "PB", "PR", "PE", "PI", "RJ", "RN", "RS", "RO", "RR", "SC", "SP", "SE", "TO")
LOCATION_PROFILES = (
    {"cep": "01001000", "uf": "SP", "city": "São Paulo", "neighborhood": "Sé", "street": "Praça da Sé", "ibge": "3550308"},
    {"cep": "20040002", "uf": "RJ", "city": "Rio de Janeiro", "neighborhood": "Centro", "street": "Rua da Assembleia", "ibge": "3304557"},
    {"cep": "30140071", "uf": "MG", "city": "Belo Horizonte", "neighborhood": "Funcionários", "street": "Avenida Afonso Pena", "ibge": "3106200"},
    {"cep": "40020000", "uf": "BA", "city": "Salvador", "neighborhood": "Centro", "street": "Avenida Sete de Setembro", "ibge": "2927408"},
    {"cep": "50030000", "uf": "PE", "city": "Recife", "neighborhood": "Santo Antônio", "street": "Rua do Imperador", "ibge": "2611606"},
    {"cep": "60025061", "uf": "CE", "city": "Fortaleza", "neighborhood": "Centro", "street": "Rua Major Facundo", "ibge": "2304400"},
    {"cep": "70040902", "uf": "DF", "city": "Brasília", "neighborhood": "Asa Sul", "street": "SBS Quadra 2", "ibge": "5300108"},
    {"cep": "80010010", "uf": "PR", "city": "Curitiba", "neighborhood": "Centro", "street": "Rua XV de Novembro", "ibge": "4106902"},
    {"cep": "90010000", "uf": "RS", "city": "Porto Alegre", "neighborhood": "Centro Histórico", "street": "Rua dos Andradas", "ibge": "4314902"},
    {"cep": "88010001", "uf": "SC", "city": "Florianópolis", "neighborhood": "Centro", "street": "Rua Felipe Schmidt", "ibge": "4205407"},
    {"cep": "74003010", "uf": "GO", "city": "Goiânia", "neighborhood": "Setor Central", "street": "Avenida Goiás", "ibge": "5208707"},
    {"cep": "69005040", "uf": "AM", "city": "Manaus", "neighborhood": "Centro", "street": "Avenida Eduardo Ribeiro", "ibge": "1302603"},
)


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


def _merged_validation(field: dict) -> dict:
    validation = dict(field.get("validation") or {})
    for key, value in field.items():
        if key in {"allowed_values", "digit_count", "pattern", "min", "max", "min_value", "max_value", "allow_digits", "format"}:
            validation.setdefault(key, value)
    return validation


def apply_shared_rules(field_map: dict) -> dict:
    shared_rules = field_map.get("shared_rules") or {}
    resolved_fields = []
    for field in field_map.get("fields", []):
        resolved = dict(field)
        validation = dict(field.get("validation") or {})
        for reference in field.get("rule_refs") or []:
            rule = shared_rules.get(reference) or {}
            for key, value in rule.items():
                if key != "description":
                    validation[key] = value
        if validation:
            resolved["validation"] = validation
        resolved_fields.append(resolved)
    return {**field_map, "fields": resolved_fields}


def _location_group(field: dict) -> str | None:
    text = _normalized_text(" ".join(str(field.get(key, "")) for key in ("field_id", "label", "description")))
    if "cartorio" in text:
        return "registry"
    if "residencia" in text or "logradouro" in text:
        return "residence"
    if "ocorrencia" in text or "endereco da ocorrencia" in text:
        return "occurrence"
    return None


def _location_profile(field: dict, rng: random.Random, context: dict | None) -> dict | None:
    if context is None:
        return None
    group = _location_group(field)
    if group is None:
        return None
    locations = context.setdefault("locations", {})
    if group not in locations:
        locations[group] = dict(rng.choice(LOCATION_PROFILES))
    return locations[group]


def _gestational_history(rng: random.Random, context: dict | None) -> dict[str, int]:
    if context is None:
        return {
            "gestacoes": rng.randint(0, 8),
            "partos_vaginais": rng.randint(0, 5),
            "cesareas": rng.randint(0, 3),
            "nascidos_vivos": rng.randint(0, 8),
            "perdas": rng.randint(0, 4),
        }
    history = context.get("gestational_history")
    if history is None:
        gestacoes = rng.choices(range(0, 9), weights=(20, 22, 18, 14, 10, 7, 4, 3, 2), k=1)[0]
        perdas = rng.randint(0, min(gestacoes, 3))
        births = max(0, gestacoes - perdas)
        cesareas = rng.randint(0, births)
        history = {
            "gestacoes": gestacoes,
            "partos_vaginais": births - cesareas,
            "cesareas": cesareas,
            "nascidos_vivos": max(0, births - rng.choice((0, 0, 0, 1)) if births else 0),
            "perdas": perdas,
        }
        context["gestational_history"] = history
    return history


def _bounded_integer(field: dict, rng: random.Random, context: dict | None = None,
                     width: int | None = None) -> int:
    validation = _merged_validation(field)
    label = _normalized_text(" ".join(str(field.get(key, "")) for key in ("field_id", "label", "description", "notes")))
    if "gestacoes anteriores" in label:
        return _gestational_history(rng, context)["gestacoes"]
    if "partos vaginais" in label:
        return _gestational_history(rng, context)["partos_vaginais"]
    if "cesareas" in label:
        return _gestational_history(rng, context)["cesareas"]
    if "nascidos vivos" in label:
        return _gestational_history(rng, context)["nascidos_vivos"]
    if "perdas fetais" in label or "abortos" in label:
        return _gestational_history(rng, context)["perdas"]
    if "apgar" in label:
        return rng.randint(int(validation.get("min", 0)), int(validation.get("max", 10)))
    if "semanas de gestacao" in label:
        return rng.randint(22, 42)
    if "consultas" in label and "pre-natal" in label:
        return rng.randint(0, 14)
    if "idade" in label:
        return rng.randint(18, 65)
    if "numero" in label or "número" in label:
        upper = min(999, 10 ** (width or 3) - 1)
        return rng.randint(1, upper)
    minimum = int(validation.get("min", validation.get("min_value", 0)))
    maximum = int(validation.get("max", validation.get("max_value", 10 ** (width or 2) - 1)))
    return rng.randint(minimum, maximum)


def _random_digit_sequence(field: dict, rng: random.Random, context: dict | None = None) -> str:
    elements = field.get("elements") or []
    width = len(elements) if elements else int(_merged_validation(field).get("digit_count", 1))
    validation = _merged_validation(field)
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
    validation = _merged_validation(field)
    pattern = validation.get("pattern")
    if not pattern or re.fullmatch(pattern, value):
        return value
    if pattern == "^[0-9]{8}$":
        return "".join(str(rng.randrange(10)) for _ in range(8))
    if validation.get("allow_digits") is False:
        return f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)} {rng.choice(LAST_NAMES)}"
    return value


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


    fine = np.asarray(
        [[rng.uniform(0.78, 1.0) for _ in range(width)] for _ in range(height)],
        dtype=np.float32,
    )
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
    grain = np.asarray(
        [[rng.uniform(0.48, 1.0) for _ in range(width)] for _ in range(height)],
        dtype=np.float32,
    )
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


def _render_glyph(character: str, height: int, style: HandwritingStyle,
                  rng: random.Random, intensity: float, medium: str = "pen") -> Image.Image:
    scale = 4
    font_size = max(12, round(height * scale * 1.08 * style.size_scale))
    font = ImageFont.truetype(str(HANDWRITING_FONT), font_size)
    bbox = font.getbbox(character or " ", stroke_width=1)
    width = max(round(height * scale * 0.20), bbox[2] - bbox[0] + 10 * scale)
    glyph = Image.new("L", (width, height * scale + 12 * scale), 0)
    draw = ImageDraw.Draw(glyph)
    draw.text(
        (glyph.width // 2, glyph.height // 2), character, font=font, fill=235,
        anchor="mm", stroke_width=max(0, 1 + style.stroke_bias), stroke_fill=220,
    )
    crop = glyph.getbbox()
    if crop is None:
        return Image.new("L", (max(1, round(height * 0.3)), height), 0)
    glyph = glyph.crop((max(0, crop[0]-4), max(0, crop[1]-4), min(glyph.width, crop[2]+4), min(glyph.height, crop[3]+4)))
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
                            intensity: float = 0.65, medium: str = "pencil") -> Image.Image:
    if not HANDWRITING_FONT.is_file():
        raise FileNotFoundError(f"Fonte manuscrita não encontrada: {HANDWRITING_FONT}")
    if medium not in {"pen", "pencil"}:
        raise ValueError(f"Material de escrita desconhecido: {medium}")
    width, height = max(1, int(width)), max(1, int(height))
    intensity = min(1.0, max(0.0, intensity))
    style = style or random_handwriting_style(rng, intensity, medium=medium)
    glyphs = [_render_glyph(character, height, style, rng, intensity, medium=medium) for character in text]
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
        mask.paste(content, (max(0, (width-content.width)//2), 0))
    rgba = Image.new("RGBA", (width, height), (*style.ink, 0))
    rgba.putalpha(mask)
    return rgba


def render_handwritten_numeric(text: str, width: int, height: int, rng: random.Random,
                               style: HandwritingStyle | None = None,
                               intensity: float = 0.65) -> Image.Image:
    if not is_numeric_text(text):
        raise ValueError("O renderizador manuscrito aceita somente números e separadores")
    return render_handwritten_text(
        text, width, height, rng, style=style, intensity=intensity, medium="pen",
    )


def _fit_mark_image(image: Image.Image, width: int, height: int, variant: str) -> Image.Image:
    canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    alpha_bbox = image.getchannel("A").getbbox()
    if alpha_bbox is not None:
        image = image.crop(alpha_bbox)
    if image.width <= 0 or image.height <= 0:
        return canvas
    if variant in {"middle_stroke", "upper_stroke"}:
        target_width, target_height = max(1, round(width * 0.90)), max(1, round(height * 0.42))
    elif variant == "x":
        target_width, target_height = max(1, round(width * 0.78)), max(1, round(height * 0.78))
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


def render_neural_text(text: str, width: int, height: int, pen: NeuralPen,
                       model, rng: random.Random) -> tuple[Image.Image, dict]:
    rendered = prepare_text(text, pen.text_policy)
    chunks = [rendered]
    if pen.model == "sjvasquez" and len(rendered) > 12:
        chunks = [rendered[index:index + 10].strip() for index in range(0, len(rendered), 10)]
        chunks = [chunk for chunk in chunks if chunk]
    canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    weights = [max(1, len(chunk)) for chunk in chunks]
    gap = max(1, round(width * 0.01)) if len(chunks) > 1 else 0
    available_width = max(len(chunks) * 4, width - gap * (len(chunks) - 1))
    cursor = 0
    seeds = []
    total_points = 0
    for index, (chunk, weight) in enumerate(zip(chunks, weights)):
        chunk_width = max(4, available_width - cursor if index == len(chunks) - 1 else round(available_width * weight / sum(weights)))
        last_error = None
        for _ in range(4):
            seed = rng.getrandbits(32)
            try:
                points = model.sample(chunk, pen.writer or 0, pen.bias, seed)
                image = render_strokes(points, chunk_width, height, pen, seed)
                break
            except ValueError as exc:
                last_error = exc
        else:
            raise ValueError(f"O modelo {pen.model} não conseguiu escrever o valor após novas tentativas") from last_error
        canvas.alpha_composite(image, (cursor + index * gap, 0))
        cursor += chunk_width
        seeds.append(seed)
        total_points += len(points)
    return canvas, {"backend": pen.model, "seeds": seeds, "stroke_points": total_points}


def render_checkbox_mark(width: int, height: int, rng: random.Random,
                         style: HandwritingStyle | None = None,
                         intensity: float = 0.65,
                         neural_pen: NeuralPen | None = None,
                         handwriting_model=None,
                         render_rng: random.Random | None = None) -> tuple[Image.Image, dict]:
    width, height = max(1, int(width)), max(1, int(height))
    variant = rng.choice(("middle_stroke", "upper_stroke", "vertical_stroke", "x"))
    character = "x" if variant == "x" else "I"
    rotation = rng.uniform(-8, 8)
    if variant in {"middle_stroke", "upper_stroke"}:
        rotation += rng.choice((-90, 90))

    if neural_pen is not None:
        if handwriting_model is None:
            handwriting_model = load_model(neural_pen.model)
        sample_rng = render_rng or rng
        rendered = prepare_text(character, neural_pen.text_policy)
        image, neural_metadata = render_neural_text(rendered, width, height, neural_pen, handwriting_model, sample_rng)
        backend = neural_pen.model
        metadata = {
            **neural_metadata, "backend": backend, "mark_variant": variant,
            "requested_value": character,
        }
    else:
        image = render_handwritten_text(
            character, width, height, rng, style=style, intensity=intensity, medium="pen",
        )
        metadata = {"backend": "font", "mark_variant": variant, "requested_value": character}
    image = image.rotate(rotation, resample=Image.Resampling.BICUBIC, expand=True)
    return _fit_mark_image(image, width, height, variant), {"rendering": metadata}


def random_value(field: dict, rng: random.Random, context: dict | None = None) -> str:
    data_type = field.get("data_type", "text")
    validation = _merged_validation(field)
    allowed = validation.get("allowed_values") or []
    label = _normalized_text(" ".join(str(field.get(key, "")) for key in ("field_id", "label", "description", "notes")))
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
            return rng.choice(("Apto 101", "Bloco B", "Casa 2", "Fundos"))
        if "endereco" in label or "logradouro" in label:
            return profile["street"]
    if allowed:
        return str(rng.choice(allowed))
    if data_type == "uf":
        return rng.choice(UF_VALUES)
    if data_type == "digit_sequence":
        return _random_digit_sequence(field, rng, context)
    if data_type == "person_name" or "nome" in label:
        return f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)} {rng.choice(LAST_NAMES)}"
    if data_type == "date" or "data" in label:
        generated = date(1980, 1, 1) + timedelta(days=rng.randrange(0, 46 * 365))
        return generated.strftime("%d/%m/%Y")
    if data_type == "time" or "hora" in label:
        return f"{rng.randrange(0, 24):02d}:{rng.randrange(0, 60):02d}"
    if data_type == "cep" or "cep" in label:
        digits = f"{rng.randrange(0, 100_000_000):08d}"
        return f"{digits[:5]}-{digits[5:]}"
    if data_type == "cns" or "cartão sus" in label or "cartao sus" in label:
        return "".join(str(rng.randrange(10)) for _ in range(15))
    if data_type == "integer" or any(word in label for word in ("nº", "numero", "número", "idade", "quantidade")):
        return str(_bounded_integer(field, rng, context))
    if data_type == "decimal" or "peso" in label:
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


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", value).strip("_")
    return cleaned or "synthetic"


def generate_document(template_path: Path, field_map: dict, selected_ids: set[str],
                      output_dir: Path, document_id: str, rng: random.Random,
                      handwriting: bool = True, handwriting_intensity: float = 0.65,
                      text_handwriting: bool = True, neural_pen: NeuralPen | None = None,
                      numeric_neural_pen: NeuralPen | None = None,
                      text_neural_pen: NeuralPen | None = None,
                      handwriting_model=None, realistic_filters: bool = False,
                      filter_preset: str = "medium",
                      style_preset_uid: str | None = None) -> SyntheticDocument:
    field_map = apply_shared_rules(field_map)
    output_dir.mkdir(parents=True, exist_ok=True)
    document = fitz.open(template_path)
    if len(document) == 0:
        document.close()
        raise ValueError("O template não contém páginas")
    if len(document) > 1:
        document.select([0])
    page = document[0]
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
    neural_pens = [pen for pen in (numeric_neural_pen, text_neural_pen) if pen is not None]
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
    handwriting_style = random_handwriting_style(rng, handwriting_intensity, medium="pen")
    pencil_style = random_handwriting_style(rng, handwriting_intensity, medium="pencil")
    generation_context: dict = {}

    def insert_value(rect_values: list[int], text: str, font_size: float | None = None,
                     pencil: bool = False, force_printed: bool = False) -> tuple[str, dict]:
        x1, y1, x2, y2 = rect_values
        rect = fitz.Rect(x1 * point_scale, y1 * point_scale, x2 * point_scale, y2 * point_scale)
        active_pen = numeric_neural_pen if is_numeric_text(text) else text_neural_pen
        neural_enabled = handwriting if is_numeric_text(text) else text_handwriting
        if (
            not force_printed
            and active_pen is not None
            and neural_enabled
        ):
            rendered = prepare_text(text, active_pen.text_policy)
            model = handwriting_models[active_pen.model]
            image, rendering = render_neural_text(rendered, x2-x1, y2-y1, active_pen, model, render_rng)
            stream = io.BytesIO()
            image.save(stream, format="PNG")
            page.insert_image(rect, stream=stream.getvalue(), overlay=True, keep_proportion=False)
            return rendered, {"requested_value": text, "rendering": rendering}
        if not force_printed and handwriting and is_numeric_text(text):
            image = render_handwritten_numeric(
                text, max(1, x2-x1), max(1, y2-y1), rng,
                style=handwriting_style, intensity=handwriting_intensity,
            )
            stream = io.BytesIO()
            image.save(stream, format="PNG")
            page.insert_image(rect, stream=stream.getvalue(), overlay=True, keep_proportion=False)
            return text, {}
        if not force_printed and text_handwriting and pencil:
            image = render_handwritten_text(
                text, max(1, x2-x1), max(1, y2-y1), rng,
                style=pencil_style, intensity=handwriting_intensity, medium="pencil",
            )
            stream = io.BytesIO()
            image.save(stream, format="PNG")
            page.insert_image(rect, stream=stream.getvalue(), overlay=True, keep_proportion=False)
            return text, {}
        size = font_size or max(5.0, min(11.0, rect.height * 0.62))
        while size >= 1.0:
            remaining = page.insert_textbox(
                rect, text, fontsize=size, fontname="helv", color=(0.05, 0.15, 0.55),
                align=fitz.TEXT_ALIGN_CENTER if len(text) == 1 else fitz.TEXT_ALIGN_LEFT,
            )
            if remaining >= 0:
                return text, {}
            size *= 0.85
        raise ValueError(f"Texto não cabe no campo: {text!r}")

    for field in field_map.get("fields", []):
        field_id = str(field["field_id"])
        if field_id not in selected_ids:
            continue
        elements = field.get("elements") or []
        if field.get("data_type") == "checkbox" and elements:
            count = 1 if field.get("selection_mode", "single") == "single" else rng.randint(1, len(elements))
            selected_elements = rng.sample(elements, count)
            element_entries = []
            for element in elements:
                checked = element in selected_elements
                mark_metadata = {}
                if checked:
                    ex1, ey1, ex2, ey2 = element["bbox"]
                    image, mark_metadata = render_checkbox_mark(
                        ex2-ex1, ey2-ey1, rng, style=handwriting_style,
                        intensity=handwriting_intensity, neural_pen=numeric_neural_pen,
                        handwriting_model=handwriting_models.get(numeric_neural_pen.model) if numeric_neural_pen else None,
                        render_rng=render_rng,
                    )
                    stream = io.BytesIO()
                    image.save(stream, format="PNG")
                    rect = fitz.Rect(*(v * point_scale for v in element["bbox"]))
                    page.insert_image(rect, stream=stream.getvalue(), overlay=True, keep_proportion=False)
                element_entries.append({**element, "checked": checked, **mark_metadata})
            value = ",".join(str(item.get("value") or item["element_id"]) for item in selected_elements)
            values[field_id] = value
            entries.append({
                "field_id": field_id, "label": field.get("label", ""), "value": value,
                "bbox": field["bbox"], "elements": element_entries,
            })
            continue
        if field.get("data_type") in {"digit_sequence", "text_sequence"} and elements:
            ordered = sorted(elements, key=lambda item: item.get("index", 10_000))
            if field.get("data_type") == "digit_sequence":
                sequence = random_value({**field, "elements": ordered}, rng, generation_context)
            else:
                sequence = "".join(rng.choice(string.ascii_uppercase) for _ in ordered)
            if len(sequence) < len(ordered):
                sequence = sequence + "".join(rng.choice(string.digits) for _ in range(len(ordered) - len(sequence)))
            sequence = sequence[:len(ordered)]
            element_entries = []
            for character, element in zip(sequence, ordered):
                rendered, metadata = insert_value(
                    element["bbox"], character,
                    pencil=field.get("data_type") == "text_sequence",
                    force_printed=bool(field.get("render_as_printed", False)),
                )
                element_entries.append({**element, "value": rendered, **metadata})
            requested_sequence = sequence
            sequence = "".join(element["value"] for element in element_entries)
            values[field_id] = sequence
            entries.append({
                "field_id": field_id, "label": field.get("label", ""), "value": sequence,
                "bbox": field["bbox"], "elements": element_entries,
                **({"requested_value": requested_sequence} if (numeric_neural_pen or text_neural_pen) is not None else {}),
            })
            continue
        value = random_value(field, rng, generation_context)
        text_types = {"text", "person_name", "enum", "uf"}
        value, metadata = insert_value(
            field["bbox"], value,
            pencil=field.get("data_type", "text") in text_types and not is_numeric_text(value),
            force_printed=bool(field.get("render_as_printed", False)),
        )
        values[field_id] = value
        entries.append({"field_id": field_id, "label": field.get("label", ""), "value": value, "bbox": field["bbox"], **metadata})

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
                     progress_callback: Callable[[int, int, SyntheticDocument], None] | None = None) -> list[SyntheticDocument]:
    if count <= 0:
        raise ValueError("A quantidade deve ser maior que zero")
    if not selected_ids:
        raise ValueError("Selecione ao menos um campo")
    if output_dir is None:
        document_type = infer_document_type_key(template_path, mapping_path)
        output_dir = build_dataset_run_dir(document_type, count)
    field_map = json.loads(mapping_path.read_text(encoding="utf-8"))
    rng = random.Random(seed)
    split_neural_pens = numeric_neural_pen is not None or text_neural_pen is not None
    if split_neural_pens:
        numeric_neural_pen = numeric_neural_pen or neural_pen
        text_neural_pen = text_neural_pen or neural_pen
    else:
        numeric_neural_pen = neural_pen
        text_neural_pen = neural_pen
    handwriting_model = {
        model: load_model(model)
        for model in {pen.model for pen in (numeric_neural_pen, text_neural_pen) if pen is not None}
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
            style_preset_uid=style_preset_uid,
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
        "handwriting": {
            **handwriting_metadata,
            "enabled": handwriting, "numbers_enabled": handwriting,
            "text_enabled": text_handwriting,
            "intensity": handwriting_intensity,
        },
        "visual_filters": {"enabled": realistic_filters, "preset": filter_preset},
        "style_preset_uid": style_preset_uid,
        "documents": [{"pdf": item.pdf_path.name, "ground_truth": item.json_path.name} for item in generated],
    }
    (output_dir / "dataset.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return generated
