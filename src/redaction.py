import os
import re

import cv2
import numpy as np

from src import pdf as pdf_module
from src.segmentation import WordRegion

Bbox = tuple[int, int, int, int]


def draw_redactions(image: np.ndarray, boxes: list[Bbox],
                     color: tuple[int, int, int], alpha: float = 1.0) -> np.ndarray:
    result = image.copy()
    if alpha >= 1.0:
        for x1, y1, x2, y2 in boxes:
            cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness=cv2.FILLED)
        return result

    overlay = image.copy()
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness=cv2.FILLED)
    cv2.addWeighted(overlay, alpha, result, 1 - alpha, 0, dst=result)
    return result


def save_crops(image: np.ndarray, words: list[WordRegion], crops_dir: str,
                padding: int = 0) -> dict[int, str]:
    os.makedirs(crops_dir, exist_ok=True)
    height, width = image.shape[:2]
    crop_paths: dict[int, str] = {}

    for word in words:
        x1, y1, x2, y2 = word.bbox
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(width, x2 + padding)
        y2 = min(height, y2 + padding)

        crop = image[y1:y2, x1:x2]
        filename = f"word_{word.word_id:04d}.png"
        cv2.imwrite(os.path.join(crops_dir, filename), crop)
        crop_paths[word.word_id] = os.path.join("crops", filename)

    return crop_paths


def _safe_slug(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", text).strip("_")
    return slug or "field"


def save_field_crops(image: np.ndarray, field_ids: list[str], field_labels: dict[str, str],
                      field_boxes: dict[str, Bbox], crops_dir: str, padding: int = 0) -> dict[str, str]:
    os.makedirs(crops_dir, exist_ok=True)
    height, width = image.shape[:2]
    crop_paths: dict[str, str] = {}

    for i, field_id in enumerate(field_ids):
        x1, y1, x2, y2 = field_boxes[field_id]
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(width, x2 + padding)
        y2 = min(height, y2 + padding)

        crop = image[y1:y2, x1:x2]
        filename = f"field_{i:04d}_{_safe_slug(field_labels.get(field_id, field_id))}.png"
        cv2.imwrite(os.path.join(crops_dir, filename), crop)
        crop_paths[field_id] = os.path.join("crops", filename)

    return crop_paths


def build_output_pdf(masked_image: np.ndarray, dpi: int, output_pdf_path: str) -> None:
    pdf_module.image_to_pdf(masked_image, dpi, output_pdf_path)
