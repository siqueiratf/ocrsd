"""Compare checkbox-mark candidates (legacy I/x vs slash/backslash/x_combo) across
preset families, at real checkbox box sizes, from the same seeds.

For each (document, preset, candidate) it renders N marks at the document's real
checkbox box size, saves a contact sheet (one row per candidate, one column per
sample), and reports geometry: mark_size_ratio, mark_center_offset_frac (vs the box
centre), mark_stroke_width_px (ink_area / max(mark_w, mark_h)), mark_angle_deg,
mark_ink_outside_frac (ink outside the box, note some real pen overshoot is allowed
but must not approach neighbouring boxes/labels - checked qualitatively in the sheets).

Usage: PYTHONPATH=. /home/ts/ocrsd/.venv/bin/python scripts/compare_checkbox_marks.py
"""
from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.document_types import discover_document_types  # noqa: E402
from src.synthetic_dataset import render_checkbox_mark, random_handwriting_style  # noqa: E402
from writing_audit import build_presets, neural_models_available, make_pen  # noqa: E402

OUT_DIR = ROOT / "var" / "t3" / "checkbox_compare"
DOCUMENTS = ["do", "dnv"]
PRESET_UIDS = [
    "padrao_fonte_natural", "padrao_fonte_homemade_apple",
    "style_20260921_144107_d6344a", "style_20260921_140241_6d9869",
]
CANDIDATES = ["legacy", "slash_set"]
SAMPLES_PER_CELL = 6
SEED_BASE = 1000


def mark_geometry(image: Image.Image, box_w: int, box_h: int) -> dict:
    alpha = np.asarray(image.getchannel("A"))
    ys, xs = np.nonzero(alpha > 30)
    if xs.size == 0:
        return {"mark_present": False}
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    mark_w, mark_h = x1 - x0 + 1, y1 - y0 + 1
    size_ratio = (mark_w * mark_h) / (box_w * box_h)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    center_offset_frac = math.hypot(cx - box_w / 2, cy - box_h / 2) / max(box_w, box_h)
    stroke_width_px = xs.size / max(mark_w, mark_h)
    outside = (xs < 0) | (xs >= box_w) | (ys < 0) | (ys >= box_h)
    ink_outside_frac = float(outside.sum()) / xs.size
    angle_deg = None
    if xs.size >= 4:
        pts = np.stack([xs - cx, ys - cy]).astype(float)
        cov = np.cov(pts)
        if cov.shape == (2, 2) and np.all(np.isfinite(cov)):
            eigvals, eigvecs = np.linalg.eigh(cov)
            principal = eigvecs[:, int(np.argmax(eigvals))]
            angle_deg = round(math.degrees(math.atan2(principal[1], principal[0])) % 180, 1)
    return {
        "mark_present": True, "mark_size_ratio": round(size_ratio, 3),
        "mark_center_offset_frac": round(center_offset_frac, 3),
        "mark_stroke_width_px": round(stroke_width_px, 2),
        "mark_angle_deg": angle_deg, "mark_ink_outside_frac": round(ink_outside_frac, 3),
    }


def representative_box(document_key: str, definition: dict) -> tuple[int, int]:
    sizes = []
    for field in definition.get("fields", []):
        if field.get("data_type") == "checkbox" and field.get("elements"):
            for element in field["elements"]:
                x1, y1, x2, y2 = element["bbox"]
                sizes.append((x2 - x1, y2 - y1))
    sizes.sort(key=lambda wh: wh[0] * wh[1])
    return sizes[len(sizes) // 2]


def render_cell(preset: dict, candidate: str, box_w: int, box_h: int, seed_base: int):
    pen = make_pen(preset)
    images, geoms = [], []
    for i in range(SAMPLES_PER_CELL):
        rng = random.Random(f"{seed_base}:{candidate}:{preset['uid']}:{i}")
        style = None if pen is not None else random_handwriting_style(rng, preset["intensity"], medium="pen")
        image, metadata = render_checkbox_mark(
            box_w, box_h, rng, style=style, intensity=preset["intensity"], neural_pen=pen,
            font_family=preset["font_family"], mark_candidate=candidate,
        )
        images.append(image)
        geom = mark_geometry(image, box_w, box_h)
        geom["mark_variant"] = metadata["rendering"].get("mark_variant")
        geoms.append(geom)
    return images, geoms


def contact_sheet(rows: list[tuple[str, list[Image.Image]]], box_w: int, box_h: int, scale: int = 4) -> Image.Image:
    pad = 6
    label_w = 220
    cell_w, cell_h = box_w * scale, box_h * scale
    width = label_w + SAMPLES_PER_CELL * (cell_w + pad)
    height = len(rows) * (cell_h + pad) + pad
    sheet = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    from PIL import ImageDraw
    draw = ImageDraw.Draw(sheet)
    for r, (label, images) in enumerate(rows):
        y = pad + r * (cell_h + pad)
        draw.text((4, y + cell_h // 2 - 6), label, fill=(0, 0, 0, 255))
        for c, image in enumerate(images):
            x = label_w + c * (cell_w + pad)
            cell = Image.new("RGBA", (cell_w, cell_h), (245, 245, 245, 255))
            draw2 = ImageDraw.Draw(cell)
            draw2.rectangle([0, 0, cell_w - 1, cell_h - 1], outline=(180, 180, 180, 255))
            big = image.resize((cell_w, cell_h), Image.Resampling.NEAREST)
            cell.alpha_composite(big)
            sheet.alpha_composite(cell, (x, y))
    return sheet.convert("RGB")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    types = discover_document_types(ROOT / "documents")
    avail = neural_models_available()
    presets = {p["uid"]: p for p in build_presets(avail)}
    report = {}
    for doc_key in DOCUMENTS:
        definition = json.loads(types[doc_key].active_mapping.read_text())
        box_w, box_h = representative_box(doc_key, definition)
        report[doc_key] = {"box": [box_w, box_h], "presets": {}}
        for uid in PRESET_UIDS:
            preset = presets[uid]
            rows = []
            geom_by_candidate = {}
            for candidate in CANDIDATES:
                images, geoms = render_cell(preset, candidate, box_w, box_h, SEED_BASE)
                rows.append((f"{candidate}", images))
                geom_by_candidate[candidate] = geoms
            sheet = contact_sheet(rows, box_w, box_h)
            sheet_path = OUT_DIR / f"{doc_key}__{uid}.png"
            sheet.save(sheet_path)
            report[doc_key]["presets"][uid] = {"sheet": str(sheet_path.relative_to(ROOT)), "geometry": geom_by_candidate}
            print(f"wrote {sheet_path}")
    (OUT_DIR / "geometry_report.json").write_text(json.dumps(report, indent=2))
    print(f"wrote {OUT_DIR / 'geometry_report.json'}")


if __name__ == "__main__":
    main()
