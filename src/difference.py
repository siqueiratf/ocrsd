from dataclasses import dataclass

import cv2
import numpy as np

from src.config import PipelineConfig


@dataclass
class DiffResult:
    diff_gray: np.ndarray
    binary_mask: np.ndarray


def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def adaptive_open_kernel(scale: float, base_kernel: tuple[int, int]) -> tuple[int, int]:
    deviation = max(scale, 1.0 / scale) - 1.0
    size = round(base_kernel[0] + 10 * (deviation ** 0.5))
    size = max(base_kernel[0], min(15, size))
    if size % 2 == 0:
        size += 1
    return (size, size)


def _drop_small_contours(mask: np.ndarray, min_area: int, min_height: int) -> np.ndarray:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cleaned = np.zeros_like(mask)
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
        _, _, _, h = cv2.boundingRect(contour)
        if h < min_height:
            continue
        cv2.drawContours(cleaned, [contour], -1, 255, thickness=cv2.FILLED)
    return cleaned


def run_difference(template_img: np.ndarray, aligned_img: np.ndarray,
                    valid_mask: np.ndarray, config: PipelineConfig,
                    open_kernel_size: tuple[int, int] | None = None) -> DiffResult:
    template_gray = _to_gray(template_img)
    aligned_gray = _to_gray(aligned_img)

    diff_gray = cv2.absdiff(template_gray, aligned_gray)
    diff_gray = cv2.bitwise_and(diff_gray, diff_gray, mask=valid_mask)

    _, binary_mask = cv2.threshold(diff_gray, config.diff_threshold, 255, cv2.THRESH_BINARY)

    open_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, open_kernel_size or config.morph_open_kernel
    )
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, open_kernel)

    binary_mask = _drop_small_contours(binary_mask, config.min_component_area, config.min_component_height)

    return DiffResult(diff_gray=diff_gray, binary_mask=binary_mask)
