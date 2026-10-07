import json
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass
class FieldElement:
    element_id: str
    bbox: tuple[int, int, int, int]
    label: str = ""
    value: str | None = None
    index: int | None = None


@dataclass
class FieldDef:
    field_id: str
    label: str
    bbox: tuple[int, int, int, int]
    description: str = ""
    data_type: str = "text"
    validation: dict[str, Any] | None = None
    selection_mode: str = "single"
    elements: list[FieldElement] | None = None


    always_redact: bool = False


@dataclass
class FieldMap:
    template_id: str
    dpi: int
    page_size_px: tuple[int, int]
    fields: list[FieldDef]


def load_field_map(path: str) -> FieldMap:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    fields = [
        FieldDef(
            field_id=str(f["field_id"]),
            label=f.get("label") or f.get("description") or str(f["field_id"]),
            bbox=tuple(f["bbox"]),
            description=f.get("description", ""),
            data_type=f.get("data_type", "text"),
            validation=f.get("validation", {}),
            selection_mode=f.get("selection_mode", "single"),
            elements=[
                FieldElement(
                    element_id=str(element["element_id"]),
                    label=element.get("label", ""),
                    value=element.get("value"),
                    index=element.get("index"),
                    bbox=tuple(element["bbox"]),
                )
                for element in f.get("elements", [])
            ],
            always_redact=f.get("always_redact", False),
        )
        for f in data["fields"]
    ]
    return FieldMap(
        template_id=data["template_id"],
        dpi=data["dpi"],
        page_size_px=tuple(data["page_size_px"]),
        fields=fields,
    )


def build_field_mask(field_map: FieldMap, shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape[:2], dtype=np.uint8)
    for field in field_map.fields:
        boxes = [element.bbox for element in field.elements] if field.elements else [field.bbox]
        for x1, y1, x2, y2 in boxes:
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, thickness=cv2.FILLED)
    return mask


def _center(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _overlap_area(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0
    return (ix2 - ix1) * (iy2 - iy1)


def field_for_bbox(bbox: tuple[int, int, int, int], field_map: FieldMap) -> FieldDef | None:
    cx, cy = _center(bbox)
    for field in field_map.fields:
        x1, y1, x2, y2 = field.bbox
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            return field

    best_field, best_overlap = None, 0
    for field in field_map.fields:
        overlap = _overlap_area(bbox, field.bbox)
        if overlap > best_overlap:
            best_field, best_overlap = field, overlap
    return best_field


def assign_components_to_fields(
    components: list[tuple[int, int, int, int]], field_map: FieldMap
) -> dict[str, list[tuple[int, int, int, int]]]:
    assigned: dict[str, list[tuple[int, int, int, int]]] = {f.field_id: [] for f in field_map.fields}
    for bbox in components:
        field = field_for_bbox(bbox, field_map)
        if field is not None:
            assigned[field.field_id].append(bbox)
    return assigned


def assign_components_to_elements(
    components: list[tuple[int, int, int, int]], field: FieldDef,
) -> dict[str, list[tuple[int, int, int, int]]]:
    elements = field.elements or []
    assigned = {element.element_id: [] for element in elements}
    for bbox in components:
        cx, cy = _center(bbox)
        containing = [
            element for element in elements
            if element.bbox[0] <= cx <= element.bbox[2] and element.bbox[1] <= cy <= element.bbox[3]
        ]
        if containing:
            assigned[containing[0].element_id].append(bbox)
            continue
        best = max(elements, key=lambda element: _overlap_area(bbox, element.bbox), default=None)
        if best is not None and _overlap_area(bbox, best.bbox) > 0:
            assigned[best.element_id].append(bbox)
    return assigned
