from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class WordRegion:
    word_id: int
    line_id: int
    bbox: tuple[int, int, int, int]


def find_components(binary_mask: np.ndarray, min_area: int) -> list[tuple[int, int, int, int]]:
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        boxes.append((x, y, x + w, y + h))
    return boxes


def group_into_words(binary_mask: np.ndarray, dilate_kernel: tuple[int, int],
                      min_area: int) -> list[WordRegion]:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, dilate_kernel)
    dilated = cv2.dilate(binary_mask, kernel)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    words = []
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        words.append(WordRegion(word_id=-1, line_id=-1, bbox=(x, y, x + w, y + h)))
    return words


def group_into_lines(words: list[WordRegion], y_gap_ratio: float) -> list[WordRegion]:
    if not words:
        return []

    heights = [w.bbox[3] - w.bbox[1] for w in words]
    median_height = float(np.median(heights)) if heights else 1.0
    gap_threshold = max(median_height * y_gap_ratio, 1.0)

    def y_center(word: WordRegion) -> float:
        return (word.bbox[1] + word.bbox[3]) / 2.0

    ordered = sorted(words, key=y_center)

    grouped = []
    line_id = 0
    prev_center = None
    for word in ordered:
        center = y_center(word)
        if prev_center is not None and (center - prev_center) > gap_threshold:
            line_id += 1
        grouped.append(WordRegion(word_id=word.word_id, line_id=line_id, bbox=word.bbox))
        prev_center = center

    return grouped


def reading_order_sort(words: list[WordRegion]) -> list[WordRegion]:
    ordered = sorted(words, key=lambda w: (w.line_id, w.bbox[0]))
    return [
        WordRegion(word_id=i, line_id=w.line_id, bbox=w.bbox)
        for i, w in enumerate(ordered)
    ]
