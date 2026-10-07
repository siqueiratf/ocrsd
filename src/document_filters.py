from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
import tempfile

import cv2
import numpy as np

from src.pdf import image_to_pdf, pdf_to_image


@dataclass(frozen=True)
class DocumentFilterPreset:
    brightness: tuple[float, float]
    contrast: tuple[float, float]
    tint: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
    gradient: tuple[float, float]
    shadow: tuple[float, float]
    noise: tuple[float, float]
    blur: tuple[float, float]
    jpeg_quality: tuple[int, int]
    rotation: tuple[float, float]


FILTER_PRESETS = {
    "light": DocumentFilterPreset(
        brightness=(-4, 5),
        contrast=(0.96, 1.04),
        tint=((0.985, 1.01), (0.99, 1.015), (0.995, 1.025)),
        gradient=(0.02, 0.07),
        shadow=(0.02, 0.08),
        noise=(0.8, 2.2),
        blur=(0.0, 0.20),
        jpeg_quality=(88, 96),
        rotation=(-0.18, 0.18),
    ),
    "medium": DocumentFilterPreset(
        brightness=(-8, 8),
        contrast=(0.91, 1.08),
        tint=((0.97, 1.01), (0.98, 1.02), (1.0, 1.045)),
        gradient=(0.05, 0.13),
        shadow=(0.05, 0.16),
        noise=(1.5, 4.0),
        blur=(0.05, 0.38),
        jpeg_quality=(78, 92),
        rotation=(-0.35, 0.35),
    ),
    "strong": DocumentFilterPreset(
        brightness=(-14, 12),
        contrast=(0.84, 1.14),
        tint=((0.95, 1.01), (0.965, 1.025), (1.005, 1.065)),
        gradient=(0.08, 0.20),
        shadow=(0.08, 0.24),
        noise=(2.5, 6.5),
        blur=(0.08, 0.58),
        jpeg_quality=(68, 86),
        rotation=(-0.55, 0.55),
    ),
}


def _preset(name: str) -> DocumentFilterPreset:
    try:
        return FILTER_PRESETS[name]
    except KeyError as exc:
        raise ValueError(f"Preset de realismo desconhecido: {name}") from exc


def _uniform(rng: random.Random, interval: tuple[float, float]) -> float:
    return rng.uniform(interval[0], interval[1])


def _linear_illumination(height: int, width: int, strength: float, rng: random.Random) -> np.ndarray:
    x = np.linspace(-1.0, 1.0, width, dtype=np.float32)
    y = np.linspace(-1.0, 1.0, height, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    angle = rng.uniform(0, np.pi * 2)
    plane = np.cos(angle) * xx + np.sin(angle) * yy
    plane = (plane - plane.min()) / max(1e-6, float(plane.max() - plane.min()))
    return 1.0 + (plane - 0.5) * strength


def _shadow_mask(height: int, width: int, strength: float, rng: random.Random) -> np.ndarray:
    mask = np.ones((height, width), dtype=np.float32)
    passes = 1 + int(rng.random() < 0.45)
    yy, xx = np.ogrid[:height, :width]
    for _ in range(passes):
        cx = rng.uniform(-0.1, 1.1) * width
        cy = rng.uniform(-0.1, 1.1) * height
        rx = rng.uniform(0.35, 0.95) * width
        ry = rng.uniform(0.15, 0.55) * height
        blob = ((xx - cx) / max(1, rx)) ** 2 + ((yy - cy) / max(1, ry)) ** 2
        softness = np.clip(1.0 - blob, 0.0, 1.0)
        mask *= 1.0 - softness * rng.uniform(strength * 0.35, strength)
    return cv2.GaussianBlur(mask, (0, 0), sigmaX=max(width, height) * 0.035)


def _vignette(height: int, width: int, rng: random.Random) -> np.ndarray:
    y = np.linspace(-1.0, 1.0, height, dtype=np.float32)
    x = np.linspace(-1.0, 1.0, width, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    distance = np.sqrt(xx * xx + yy * yy)
    strength = rng.uniform(0.015, 0.075)
    return np.clip(1.0 - distance * strength, 0.88, 1.0)


def _rotate_keep_size(image: np.ndarray, degrees: float) -> np.ndarray:
    if abs(degrees) < 0.01:
        return image
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), degrees, 1.0)
    return cv2.warpAffine(
        image, matrix, (width, height), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255),
    )


def apply_document_filters(
    image_bgr: np.ndarray,
    rng: random.Random | None = None,
    preset: str = "medium",
) -> tuple[np.ndarray, dict]:
    rng = rng or random.Random()
    settings = _preset(preset)
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("A imagem precisa estar em BGR com 3 canais")
    image = image_bgr.astype(np.float32)
    height, width = image.shape[:2]
    metadata = {
        "enabled": True,
        "preset": preset,
        "brightness": round(_uniform(rng, settings.brightness), 4),
        "contrast": round(_uniform(rng, settings.contrast), 4),
        "tint_bgr": [round(_uniform(rng, channel), 4) for channel in settings.tint],
        "gradient_strength": round(_uniform(rng, settings.gradient), 4),
        "shadow_strength": round(_uniform(rng, settings.shadow), 4),
        "noise_sigma": round(_uniform(rng, settings.noise), 4),
        "blur_sigma": round(_uniform(rng, settings.blur), 4),
        "jpeg_quality": rng.randint(*settings.jpeg_quality),
        "rotation_degrees": round(_uniform(rng, settings.rotation), 4),
    }

    image = (image - 127.5) * metadata["contrast"] + 127.5 + metadata["brightness"]
    image *= np.asarray(metadata["tint_bgr"], dtype=np.float32).reshape(1, 1, 3)
    illumination = _linear_illumination(height, width, metadata["gradient_strength"], rng)
    illumination *= _shadow_mask(height, width, metadata["shadow_strength"], rng)
    illumination *= _vignette(height, width, rng)
    image *= illumination[:, :, None]

    if metadata["noise_sigma"] > 0:
        noise_rng = np.random.default_rng(rng.getrandbits(64))
        noise = noise_rng.normal(0, metadata["noise_sigma"], image.shape).astype(np.float32)
        image += noise
    output = np.clip(image, 0, 255).astype(np.uint8)
    if metadata["blur_sigma"] > 0.01:
        output = cv2.GaussianBlur(output, (0, 0), sigmaX=metadata["blur_sigma"])
    output = _rotate_keep_size(output, metadata["rotation_degrees"])
    ok, encoded = cv2.imencode(".jpg", output, [cv2.IMWRITE_JPEG_QUALITY, metadata["jpeg_quality"]])
    if ok:
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if decoded is not None:
            output = decoded
    return output, metadata


def apply_realistic_filters_to_pdf(
    input_pdf: Path,
    output_pdf: Path | None = None,
    *,
    dpi: int = 300,
    rng: random.Random | None = None,
    preset: str = "medium",
) -> dict:
    output_pdf = output_pdf or input_pdf
    rendered = pdf_to_image(str(input_pdf), dpi)
    filtered, metadata = apply_document_filters(rendered.image, rng=rng, preset=preset)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    if output_pdf == input_pdf:
        with tempfile.NamedTemporaryFile(suffix=".pdf", dir=str(output_pdf.parent), delete=False) as temporary:
            temporary_path = Path(temporary.name)
        try:
            image_to_pdf(filtered, dpi, str(temporary_path))
            temporary_path.replace(output_pdf)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()
    else:
        image_to_pdf(filtered, dpi, str(output_pdf))
    return metadata
