from dataclasses import dataclass

import numpy as np

from src.config import PipelineConfig
from src.segmentation import WordRegion

_reader_cache: dict[tuple, object] = {}
_backend_cache: dict[str, object] = {}


@dataclass
class OcrResult:
    text: str
    confidence: float


def _get_easyocr_reader(languages: list[str]):
    key = tuple(languages)
    if key not in _reader_cache:
        import easyocr

        _reader_cache[key] = easyocr.Reader(languages, gpu=False, verbose=False)
    return _reader_cache[key]


def crop_region(image: np.ndarray, bbox: tuple[int, int, int, int], padding: int = 0) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    height, width = image.shape[:2]
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(width, x2 + padding)
    y2 = min(height, y2 + padding)
    return image[y1:y2, x1:x2]


def build_line_strip(image: np.ndarray, words_in_line: list[WordRegion], padding: int = 4) -> np.ndarray:
    x1 = min(w.bbox[0] for w in words_in_line)
    y1 = min(w.bbox[1] for w in words_in_line)
    x2 = max(w.bbox[2] for w in words_in_line)
    y2 = max(w.bbox[3] for w in words_in_line)
    return crop_region(image, (x1, y1, x2, y2), padding)


def _recognize_easyocr(image: np.ndarray, config: PipelineConfig) -> OcrResult:
    if image.size == 0:
        return OcrResult(text="", confidence=0.0)

    reader = _get_easyocr_reader(config.ocr_languages)
    detections = reader.readtext(image, detail=1, paragraph=False)

    if not detections:
        return OcrResult(text="", confidence=0.0)

    texts = [text for _, text, _ in detections]
    confidences = [float(conf) for _, _, conf in detections]
    return OcrResult(text=" ".join(texts).strip(), confidence=sum(confidences) / len(confidences))


def recognize_word(crop: np.ndarray, config: PipelineConfig) -> OcrResult:
    if config.ocr_engine == "easyocr":
        return _recognize_easyocr(crop, config)
    from src.ocr_bench import BACKEND_FACTORIES

    if config.ocr_engine not in BACKEND_FACTORIES:
        raise ValueError(f"Unsupported OCR engine: {config.ocr_engine}")
    if crop.size == 0:
        return OcrResult(text="", confidence=0.0)
    if config.ocr_engine not in _backend_cache:
        _backend_cache[config.ocr_engine] = BACKEND_FACTORIES[config.ocr_engine]()
    prediction = _backend_cache[config.ocr_engine].recognize(crop)
    return OcrResult(text=prediction.text, confidence=prediction.confidence or 0.0)


def recognize_line(strip: np.ndarray, config: PipelineConfig) -> OcrResult:
    return recognize_word(strip, config)
