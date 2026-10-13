from __future__ import annotations

import math
import re
import subprocess
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, median

import cv2
import numpy as np
import pymupdf

from src.mapping_assist import geometry as geo
from src.mapping_assist import ocr
from src.mapping_assist.geometry import Box

DPI = 300
SCALE = DPI / 72
MAX_DPI = 450
OCR_ERRORS = (OSError, RuntimeError, subprocess.SubprocessError)
NUMBER = re.compile(r"\d{1,3}|[a-c]")
ROMAN = ((1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"), (90, "XC"), (50, "L"), (40, "XL"), (10, "X"),
         (9, "IX"), (5, "V"), (4, "IV"), (1, "I"))


@dataclass(frozen=True)
class FieldCandidate:
    id: str
    label: str
    label_bbox: tuple[int, int, int, int] | None
    region: tuple[int, int, int, int]
    kind: str
    elements: tuple[dict, ...] = ()
    section: str | None = None
    section_title: str | None = None
    neighbors: tuple[str, ...] = ()
    number: str | None = None
    source: str = "text_layer"
    confidence: float | None = None


@dataclass(frozen=True)
class PageAnalysis:
    page: int
    dpi: int
    page_size_px: tuple[int, int]
    unit: float
    source: str
    candidates: tuple[FieldCandidate, ...]
    warnings: tuple[str, ...] = ()
    timings: dict = field(default_factory=dict)


@dataclass
class _Work:
    gray: np.ndarray
    dpi: float
    transform: tuple[float, float, float, float]
    source: str
    words: list[dict]
    vector: bool = False
    upscale: tuple[int, float] | None = None


@dataclass(eq=False)
class _Square:
    box: Box
    kind: str
    number: str | None = None
    code: str | None = None
    label: str = ""
    label_box: Box | None = None


@dataclass(eq=False)
class _Block:
    lines: list[dict]
    anchor: _Square | None = None
    cell: Box | None = None
    role: str = "label"
    used: bool = False
    owner: _Block | None = None

    @property
    def box(self) -> Box:
        return geo.union([line["box"] for line in self.lines])

    @property
    def left(self) -> float:
        return min(self.box[0], self.anchor.box[0]) if self.anchor else self.box[0]

    @property
    def text(self) -> str:
        text = re.sub(r"\s+", " ", " ".join(line["text"] for line in self.lines)).strip()
        return re.sub(r"^[^\w(]+|[\s:;,.\-–]+$", "", text)

    @property
    def size(self) -> float:
        sizes = [w["size"] for line in self.lines for w in line["words"] if w.get("size")]
        return median(sizes) if sizes else median(w["box"][3] - w["box"][1] for line in self.lines for w in line["words"])

    @property
    def confidence(self) -> float | None:
        values = [w["conf"] for line in self.lines for w in line["words"] if w.get("conf") is not None]
        return round(mean(values) / 100, 3) if values else None


@dataclass(eq=False)
class _Draft:
    kind: str
    region: Box
    block: _Block | None = None
    elements: list[dict] = field(default_factory=list)
    anchor: _Square | None = None


def roman(number: int) -> str:
    out = ""
    for value, symbol in ROMAN:
        count, number = divmod(number, value)
        out += symbol * count
    return out


def _open(pdf: Path | str | bytes) -> pymupdf.Document:
    if isinstance(pdf, (bytes, bytearray, memoryview)):
        return pymupdf.open(stream=bytes(pdf), filetype="pdf")
    return pymupdf.open(str(pdf))


def page_count(pdf: Path | str | bytes) -> int:
    with _open(pdf) as doc:
        return doc.page_count


def _pixmap_gray(pix: pymupdf.Pixmap) -> np.ndarray:
    if pix.alpha:
        pix = pymupdf.Pixmap(pix, 0)
    if pix.n != 1:
        pix = pymupdf.Pixmap(pymupdf.csGRAY, pix)
    return np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.stride)[:, :pix.width].copy()


def _render(pg: pymupdf.Page) -> np.ndarray:
    return _pixmap_gray(pg.get_pixmap(dpi=DPI, colorspace=pymupdf.csGRAY, alpha=False))


def _text_words(pg: pymupdf.Page) -> list[dict]:
    blocks = pg.get_text("dict", flags=pymupdf.TEXTFLAGS_DICT & ~pymupdf.TEXT_PRESERVE_IMAGES)["blocks"]
    spans = [(span["bbox"], span["size"], line["dir"][0] > 0.9) for block in blocks for line in block.get("lines", []) for span in line["spans"]]
    words = []
    for x0, y0, x1, y1, text, *_ in pg.get_text("words"):
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        size, horizontal = next(((s, h) for b, s, h in spans if b[0] - 0.5 <= cx <= b[2] + 0.5 and b[1] - 0.5 <= cy <= b[3] + 0.5), (None, True))
        words.append({"text": text, "conf": None, "size": size, "horizontal": horizontal,
                      "box": (x0 * SCALE, y0 * SCALE, x1 * SCALE, y1 * SCALE)})
    return words


def _dark(color) -> bool:
    if color is None:
        return False
    values = list(color) if isinstance(color, (list, tuple)) else [color]
    if len(values) == 4:
        c, m, y, k = values
        values = [(1 - c) * (1 - k), (1 - m) * (1 - k), (1 - y) * (1 - k)]
    if len(values) == 1:
        values *= 3
    return 0.299 * values[0] + 0.587 * values[1] + 0.114 * values[2] < 0.6


def _bezier(p0, p1, p2, p3, steps: int = 8) -> list[tuple[float, float]]:
    return [((1 - t) ** 3 * p0.x + 3 * (1 - t) ** 2 * t * p1.x + 3 * (1 - t) * t ** 2 * p2.x + t ** 3 * p3.x,
             (1 - t) ** 3 * p0.y + 3 * (1 - t) ** 2 * t * p1.y + 3 * (1 - t) * t ** 2 * p2.y + t ** 3 * p3.y)
            for t in np.linspace(0, 1, steps)]


def _paths(items: list) -> list[tuple[list[tuple[float, float]], bool]]:
    shapes: list[tuple[list[tuple[float, float]], bool]] = []
    current: list[tuple[float, float]] = []
    for item in items:
        if item[0] in ("re", "qu"):
            q = item[1].quad if item[0] == "re" else item[1]
            shapes.append(([tuple(q.ul), tuple(q.ur), tuple(q.lr), tuple(q.ll)], True))
            continue
        points = [tuple(item[1]), tuple(item[2])] if item[0] == "l" else _bezier(*item[1:5])
        if current and math.dist(current[-1], points[0]) < 0.05:
            current += points[1:]
        else:
            if current:
                shapes.append((current, False))
            current = list(points)
    if current:
        shapes.append((current, False))
    return shapes


def _vector_gray(pg: pymupdf.Page, size: tuple[int, int]) -> np.ndarray:
    mask = np.zeros((size[1], size[0]), np.uint8)
    for drawing in pg.get_drawings():
        kind = drawing.get("type") or ""
        filled = "f" in kind and _dark(drawing.get("fill")) and (drawing.get("fill_opacity") or 1) > 0.5
        stroked = "s" in kind and _dark(drawing.get("color")) and (drawing.get("stroke_opacity") or 1) > 0.5
        if not (filled or stroked):
            continue
        width = max(1, int(round((drawing.get("width") or 0.5) * SCALE)))
        shapes = _paths(drawing["items"])
        if filled:
            polygon = [np.round(np.array(points) * SCALE).astype(np.int32) for points, _ in shapes if len(points) >= 3]
            if polygon:
                cv2.fillPoly(mask, polygon, 255)
        if stroked:
            for points, closed in shapes:
                cv2.polylines(mask, [np.round(np.array(points) * SCALE).astype(np.int32)], closed or bool(drawing.get("closePath")), 255, width)
    return 255 - mask


def _load(doc: pymupdf.Document, pg: pymupdf.Page, size: tuple[int, int], warnings: list[str]) -> _Work:
    words = _text_words(pg) if not pg.rotation else []
    page_area = pg.rect.width * pg.rect.height
    images = [i for i in pg.get_image_info(xrefs=True) if pymupdf.Rect(i["bbox"]).get_area() >= 0.3 * page_area]
    if sum(ocr.is_alpha(w["text"]) and len(w["text"]) >= 2 for w in words) >= (15 if images else 3):
        gray = _vector_gray(pg, size)
        h, v = geo.line_masks(geo.binarize(gray, 128), 0.25 * DPI, 0.02 * DPI)
        segments = cv2.connectedComponents(h)[0] + cv2.connectedComponents(v)[0] - 2
        vector = segments >= 4
        return _Work(gray if vector else _render(pg), DPI, (0.0, 0.0, 1.0, 1.0), "text_layer", words, vector=vector)
    if pg.rotation:
        warnings.append(f"A página está rotacionada em {pg.rotation}°; a análise usou a página renderizada.")
    if len(images) == 1 and not pg.rotation and images[0].get("xref"):
        image = images[0]
        a, b, c, d, _, _ = image["transform"]
        box = pymupdf.Rect(image["bbox"])
        width, height = image["width"], image["height"]
        dpi_x, dpi_y = width / (box.width / 72), height / (box.height / 72)
        low = min(dpi_x, dpi_y)
        if abs(b) < 1e-6 and abs(c) < 1e-6 and a > 0 and d > 0 and (low >= DPI or low < 150):
            try:
                gray = _pixmap_gray(pymupdf.Pixmap(doc, image["xref"]))
            except (RuntimeError, ValueError):
                gray = None
            if gray is not None and gray.shape == (height, width):
                factor = 1 if low >= DPI else 3 if low < 100 else 2
                if low > MAX_DPI:
                    gray = cv2.resize(gray, (round(width * DPI / dpi_x), round(height * DPI / dpi_y)), interpolation=cv2.INTER_AREA)
                    dpi_x = dpi_y = DPI
                elif factor > 1:
                    gray = cv2.resize(gray, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)
                transform = (box.x0 * SCALE, box.y0 * SCALE, box.width * SCALE / gray.shape[1], box.height * SCALE / gray.shape[0])
                return _Work(gray, (dpi_x + dpi_y) / 2 * factor, transform, "raster", [], upscale=(factor, low) if factor > 1 else None)
    return _Work(_render(pg), DPI, (0.0, 0.0, 1.0, 1.0), "raster", [])


def _read_numbers(gray: np.ndarray, boxes: list[Box]) -> dict[int, str]:
    height, gap = 48, 36
    tiles = []
    for x1, y1, x2, y2 in boxes:
        mx, my = 0.14 * (x2 - x1), 0.14 * (y2 - y1)
        crop = gray[int(y1 + my):math.ceil(y2 - my), int(x1 + mx):math.ceil(x2 - mx)]
        crop = crop if crop.size else np.full((height, height), 255, np.uint8)
        tiles.append(cv2.resize(crop, (max(8, int(crop.shape[1] * height / crop.shape[0])), height), interpolation=cv2.INTER_CUBIC))
    canvas = np.full((len(tiles) * (height + gap) + gap, max(t.shape[1] for t in tiles) + 2 * gap), 255, np.uint8)
    for index, tile in enumerate(tiles):
        y = gap + index * (height + gap)
        canvas[y:y + height, gap:gap + tile.shape[1]] = tile
    found: dict[int, str] = {}
    for word in ocr.run_tesseract(canvas, psm=6, options=("-c", "tessedit_char_whitelist=0123456789abc")):
        index = int((geo.center(word["box"])[1] - gap / 2) // (height + gap))
        if 0 <= index < len(tiles):
            found[index] = found.get(index, "") + word["text"]
    return {i: text for i, text in found.items() if NUMBER.fullmatch(text)}


def _read_titles(gray: np.ndarray, bars: list[Box], unit: float) -> list[str]:
    scale = min(3.0, max(1.0, 32 / unit))
    pieces = []
    for x1, y1, x2, y2 in bars:
        crop = 255 - gray[int(y1) + 2:int(y2) - 2, int(x1) + 2:int(x2) - 2]
        crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
        pieces.append(cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC) if scale > 1 else crop)
    canvas = np.full((sum(p.shape[0] + 40 for p in pieces) + 40, max(p.shape[1] for p in pieces) + 60), 255, np.uint8)
    spans, y = [], 40
    for piece in pieces:
        canvas[y:y + piece.shape[0], 30:30 + piece.shape[1]] = piece
        spans.append((y - 20, y + piece.shape[0] + 20))
        y += piece.shape[0] + 40
    titles = [[] for _ in bars]
    for word in ocr.run_tesseract(canvas, psm=6):
        text, cy = ocr.clean_text(word["text"]), geo.center(word["box"])[1]
        index = next((i for i, (a, b) in enumerate(spans) if a <= cy <= b), None)
        if index is not None and sum(c.isalpha() for c in text) >= 2 and word["conf"] >= 60:
            titles[index].append(text)
    return [" ".join(t) if sum(len(w) for w in t) >= 4 else "" for t in titles]


class _Analyzer:
    def __init__(self, work: _Work, use_ocr: bool, warnings: list[str], timings: dict):
        self.work, self.use_ocr, self.warnings, self.timings = work, use_ocr, warnings, timings
        self.gray, self.dpi = work.gray, work.dpi
        self.ocr_ok = work.source == "text_layer"

    def _page_words(self) -> list[dict]:
        if self.work.source == "text_layer":
            return self.work.words
        if not self.use_ocr:
            self.warnings.append("OCR desativado: candidatos gerados apenas pela geometria, sem rótulos.")
            return []
        started = time.perf_counter()
        try:
            words = ocr.run_tesseract(self.gray, psm=6, dpi=self.dpi)
            self.ocr_ok = True
            if self.work.upscale:
                factor, low = self.work.upscale
                self.warnings.append(f"Imagem de baixa resolução (~{low:.0f} DPI): o OCR usou ampliação {factor}×; rótulos podem conter erros.")
            return words
        except FileNotFoundError:
            self.warnings.append("Tesseract não encontrado: candidatos gerados apenas pela geometria, sem rótulos.")
        except OCR_ERRORS as error:
            self.warnings.append(f"Falha ao executar o Tesseract ({error}): candidatos gerados apenas pela geometria, sem rótulos.")
        finally:
            self.timings["ocr"] = round(time.perf_counter() - started, 3)
        return []

    def run(self) -> tuple[list[_Draft], float]:
        gray, dpi = self.gray, self.dpi
        self.bw, self.dark = geo.binarize(gray, 185), geo.binarize(gray, 140)
        words = self._page_words()
        started = time.perf_counter()
        self.unit = unit = ocr.unit(words, 0.62 if self.work.source == "text_layer" else 1.0) or self._hole_unit() or 0.05 * dpi
        bridge = 0.02 * dpi
        h0, v0 = geo.line_masks(self.bw, 0.03 * dpi if self.work.vector else min(0.25 * dpi, 2.5 * unit), bridge, max(bridge, 1.2 * unit))
        self.v0 = v0
        self.h, self.v = geo.complete_lines(h0, v0, min(0.3 * dpi, 3 * unit), min(0.9 * dpi, 6 * unit), min(0.15 * dpi, unit), max(2, int(0.15 * unit)),
                                           self.bw)
        xs, ys = np.flatnonzero(((self.h > 0) | (self.v > 0)).any(axis=0)), np.flatnonzero(((self.h > 0) | (self.v > 0)).any(axis=1))
        self.form = form = (xs[0] - unit, ys[0] - unit, xs[-1] + unit, ys[-1] + unit) if xs.size else (0, 0, gray.shape[1], gray.shape[0])
        self.squares = [s for s in self._squares(words) if geo.holds(form, geo.center(s.box))]
        self.cells = [c for c in geo.cells(self.h, self.v, 0.015 * dpi, unit) if geo.area(c) >= (1.5 * unit) ** 2
                      and not any(geo.iou(c, s.box) >= 0.3 for s in self.squares) and self._clear_ring(c)]
        self.runs = [elements for whole in geo.comb_runs(self.bw, h0, dpi, unit) for run in geo.split_run(whole, self.v, unit)
                     if (elements := geo.run_elements(run, dpi)) and max(geo.ink(self.dark, e, 0.6) for e in elements) < 0.15
                     and not any(geo.contained(element, square.box) >= 0.3
                                 for element in elements for square in self.squares if square.kind == "checkbox")]
        self.bars = geo.bars(gray, unit, xs[0] + 0.12 * (xs[-1] - xs[0]) if xs.size else 0.15 * gray.shape[1])
        self.timings["geometry"] = round(time.perf_counter() - started, 3)
        started = time.perf_counter()
        numbers = [s for s in self.squares if s.kind == "number"]
        self.titles = [""] * len(self.bars)
        with ThreadPoolExecutor(max_workers=2) as pool:
            jobs = {}
            if self.work.source == "raster" and self.ocr_ok:
                if numbers:
                    jobs["numbers"] = pool.submit(_read_numbers, gray, [s.box for s in numbers])
                if self.bars:
                    jobs["titles"] = pool.submit(_read_titles, gray, self.bars, unit)
            words = self._words(words)
            for name, job in jobs.items():
                try:
                    result = job.result()
                except OCR_ERRORS as error:
                    what = "dos números dos campos" if name == "numbers" else "dos títulos das seções"
                    self.warnings.append(f"Falha no OCR {what} ({type(error).__name__}): essas informações ficaram sem leitura.")
                    continue
                if name == "numbers":
                    for index, text in result.items():
                        numbers[index].number = text
                else:
                    self.titles = result
        drafts = self._layout(words)
        self.timings["layout"] = round(time.perf_counter() - started, 3)
        return drafts, unit

    def _clear_ring(self, cell: Box) -> bool:
        grow = max(3, int(0.015 * self.dpi))
        line = grow + geo.line_width(self.h, self.v, cell, grow + int(0.6 * self.unit))
        return geo.ring_ink(self.bw, cell, 0.2 * self.unit + line, 0.6 * self.unit + line) < 0.5

    def _hole_unit(self) -> float | None:
        heights = [b[3] - b[1] for b in geo.rect_holes(self.bw, 0.03 * self.dpi, 0.4 * self.dpi) if 0.5 <= (b[2] - b[0]) / (b[3] - b[1]) <= 2.3]
        return median(heights) / 1.3 if len(heights) >= 3 else None

    def cell_of(self, box: Box) -> Box | None:
        point = geo.center(box)
        inside = [c for c in self.cells if geo.holds(c, point)]
        return min(inside, key=geo.area) if inside else None

    def _squares(self, words: list[dict]) -> list[_Square]:
        unit, out = self.unit, []
        for box in geo.rect_holes(self.bw, 0.6 * unit, 2.2 * unit):
            width, height = box[2] - box[0], box[3] - box[1]
            if not 0.5 <= width / height <= 2.3 or max(width, height) < unit or geo.frame(self.bw, box) < 0.8:
                continue
            line = geo.stroke(self.bw, box, int(unit))
            if geo.ring_ink(self.bw, box, 0.25 * unit + line, 0.6 * unit + line) > 0.35:
                continue
            inside = [w for w in words if geo.contained(w["box"], box) >= 0.5] if self.work.source == "text_layer" else []
            if geo.ink(self.dark, box, 0.5) < 0.05 and not inside:
                out.append(_Square(box, "checkbox"))
            elif min(width, height) >= 0.8 * unit and 0.6 <= width / height <= 1.9:
                text = re.sub(r"[^0-9a-z]", "", "".join(w["text"] for w in inside))
                out.append(_Square(box, "number", text if NUMBER.fullmatch(text) else None))
        return out

    def _blocked(self, a: Box, b: Box) -> bool:
        x1, x2 = int(min(a[2], b[2])), int(max(a[0], b[0]))
        y1, y2 = int(max(a[1], b[1])), int(min(a[3], b[3]))
        if x2 > x1 and y2 > y1 and self.v0[y1:y2, x1:x2].any():
            return True
        gap = (min(a[2], b[2]), min(a[1], b[1]), max(a[0], b[0]), max(a[3], b[3]))
        return any(geo.intersection(s.box, gap) > 0 for s in self.squares)

    def _wall(self, a: Box, b: Box) -> bool:
        x1, x2, y1, y2 = min(a[2], b[2]), max(a[0], b[0]), min(a[1], b[1]), max(a[3], b[3])
        if x2 - x1 >= 1:
            crop = self.v[int(y1):math.ceil(y2), int(x1):math.ceil(x2)]
            if crop.size and (crop > 0).mean(axis=0).max() >= 0.9:
                return True
        x1, x2, y1, y2 = min(a[0], b[0]), max(a[2], b[2]), min(a[3], b[3]), max(a[1], b[1])
        crop = self.h[int(y1):math.ceil(y2), int(x1):math.ceil(x2)] if y2 - y1 >= 1 else np.zeros((0, 0))
        return bool(crop.size) and (crop > 0).mean(axis=1).max() >= 0.9

    def _anchor_between(self, a: Box, b: Box) -> bool:
        left, right = (a, b) if a[0] <= b[0] else (b, a)
        top, bottom = min(a[1], b[1]), max(a[3], b[3])
        return any(s.kind == "number" and left[2] - 0.2 * self.unit < geo.center(s.box)[0] < right[0] + 0.2 * self.unit
                   and top - 5 * self.unit <= s.box[1] and s.box[3] <= bottom for s in self.squares)

    def _rule_between(self, upper: Box, lower: Box) -> bool:
        x1, x2 = int(max(upper[0], lower[0])), int(min(upper[2], lower[2]))
        y1, y2 = int(upper[3]), int(lower[1]) + 1
        return x2 > x1 and y2 > y1 and bool(self.h[y1:y2, x1:x2].any())

    def _words(self, words: list[dict]) -> list[dict]:
        unit, out = self.unit, []
        self.bar_words: list[list[dict]] = [[] for _ in self.bars]
        for word in words:
            text = ocr.clean_text(word["text"])
            weak = word.get("conf") is not None and word["conf"] < 30
            box = self._trim(word["box"]) if word["box"][3] - word["box"][1] > 2 * unit else word["box"]
            point = geo.center(box)
            if not text or (weak and (word["conf"] < 10 or len(text) < 3 or not text.isalpha())
                            and not (len(text) >= 2 and text.isalpha() and self._anchored(box))):
                continue
            bar = next((i for i, b in enumerate(self.bars) if geo.holds(b, point)), None)
            if bar is not None:
                self.bar_words[bar].append({**word, "text": text})
                continue
            if not word.get("horizontal", True):
                continue
            hit = next((s for s in self.squares if geo.contained(box, s.box) >= 0.5 or (s.box[0] <= point[0] <= s.box[2]
                        and geo.y_overlap(box, s.box) >= 0.3 * min(box[3] - box[1], s.box[3] - s.box[1]))), None)
            if hit is not None:
                digits = re.sub(r"[^0-9a-z]", "", text)
                if hit.kind == "number" and hit.number is None and NUMBER.fullmatch(digits):
                    hit.number = digits
                code = re.match(r"(\d{1,2})(?!\d)", word["text"])
                if hit.kind == "checkbox" and hit.code is None and code and box[0] < hit.box[0] - 0.2 * unit:
                    hit.code = code.group(1)
                continue
            for square in self.squares:
                if box[0] < square.box[2] < box[2] - 0.5 * unit and geo.y_overlap(box, square.box) > 0:
                    box = (square.box[2] + 1, box[1], box[2], box[3])
                    code = re.match(r"\W?(\d{1,2})\W", word["text"])
                    if square.kind == "checkbox" and code and square.code is None and box[0] - square.box[0] > 0:
                        square.code = code.group(1)
            out.append({**word, "text": text, "box": box, "weak": weak})
        return out

    def _trim(self, box: Box) -> Box:
        x1, y1, x2, y2 = (int(v) for v in box)
        rows = ((self.dark[y1:y2, x1:x2] > 0) & (self.h[y1:y2, x1:x2] == 0) & (self.v[y1:y2, x1:x2] == 0)).any(axis=1)
        ink = np.flatnonzero(rows)
        if not ink.size:
            return box
        gaps = np.flatnonzero(np.diff(ink) > max(2, 0.3 * self.unit))
        end = ink[gaps[0]] if gaps.size else ink[-1]
        return box[0], float(y1 + ink[0]), box[2], float(y1 + end + 1)

    def _options(self, words: list[dict]) -> set[int]:
        unit, used = self.unit, set()
        checks = sorted((s for s in self.squares if s.kind == "checkbox"), key=lambda s: s.box[0])
        digit = [bool(re.fullmatch(r"\d{1,2}", w["text"])) and not w["weak"] for w in words]
        for s in (s for s in checks if s.code is None):
            left = [i for i, w in enumerate(words) if digit[i] and i not in used and w["box"][0] < s.box[0]
                    and s.box[0] - 1.2 * unit <= w["box"][2] <= s.box[0] + 0.5 * (s.box[2] - s.box[0])
                    and geo.y_overlap(w["box"], s.box) >= 0.3 * min(s.box[3] - s.box[1], w["box"][3] - w["box"][1])]
            if left:
                index = max(left, key=lambda i: words[i]["box"][2])
                s.code = words[index]["text"]
                used.add(index)
        for s in checks:
            if s.code is None:
                below = [i for i, w in enumerate(words) if digit[i] and i not in used and s.box[3] - 0.2 * unit <= w["box"][1] <= s.box[3] + 1.2 * unit
                         and s.box[0] - 0.5 * unit <= geo.center(w["box"])[0] <= s.box[2] + 0.5 * unit]
                if below:
                    index = min(below, key=lambda i: words[i]["box"][1])
                    s.code = words[index]["text"]
                    used.add(index)
        for s in checks:
            cy = geo.center(s.box)[1]
            right = [i for i, w in enumerate(words) if i not in used and ocr.is_alpha(w["text"]) and abs(geo.center(w["box"])[1] - cy) <= 0.6 * unit
                     and s.box[2] - 0.3 * unit <= w["box"][0] <= s.box[2] + 1.5 * unit]
            if right:
                chain = self._chain(words, min(right, key=lambda i: words[i]["box"][0]), used)
                below = [i for i, w in enumerate(words) if i not in used and i not in chain and ocr.is_alpha(w["text"])
                         and abs(w["box"][0] - words[chain[0]]["box"][0]) <= 0.6 * unit
                         and 0 <= w["box"][1] - max(words[j]["box"][3] for j in chain) <= 0.6 * unit
                         and not any(t.box[2] <= w["box"][0] <= t.box[2] + 1.5 * unit and geo.y_overlap(t.box, w["box"]) > 0 for t in checks)]
                if below:
                    chain += self._chain(words, min(below, key=lambda i: words[i]["box"][0]), used | set(chain))
            elif not any(i not in used and ocr.is_alpha(w["text"]) and geo.y_overlap(w["box"], s.box) > 0 and 0 <= w["box"][0] - s.box[2] <= 3 * unit
                         for i, w in enumerate(words)):
                above = [i for i, w in enumerate(words) if i not in used and ocr.is_alpha(w["text"]) and not w["weak"] and not self._anchored(w["box"])
                         and s.box[1] - 1.6 * unit <= w["box"][3] <= s.box[1] + 0.2 * unit
                         and geo.x_overlap(w["box"], (s.box[0] - unit, 0, s.box[2] + unit, 0)) > 0]
                chain = [max(above, key=lambda i: words[i]["box"][3])] if above else []
            else:
                chain = []
            if chain:
                used.update(chain)
                s.label = " ".join(words[i]["text"] for i in chain)
                s.label_box = geo.union([words[i]["box"] for i in chain])
        return used

    def _anchored(self, box: Box) -> bool:
        return any(s.kind == "number" and s.box[2] - 0.5 * self.unit <= box[0] <= s.box[2] + 1.5 * self.unit and geo.y_overlap(box, s.box) > 0
                   for s in self.squares)

    def _chain(self, words: list[dict], start: int, used: set[int]) -> list[int]:
        unit, chain = self.unit, [start]
        cy = geo.center(words[start]["box"])[1]
        for i in sorted(range(len(words)), key=lambda i: words[i]["box"][0]):
            w, last = words[i], words[chain[-1]]["box"]
            if i in used or i in chain or w["box"][0] < last[0] or abs(geo.center(w["box"])[1] - cy) > 0.5 * unit:
                continue
            if -0.3 * unit < w["box"][0] - last[2] <= unit and not self._blocked(last, w["box"]):
                chain.append(i)
        return chain

    def _blocks(self, words: list[dict]) -> list[_Block]:
        unit = self.unit
        lines = ocr.group_lines(words, unit, self._blocked)
        for line in lines:
            line["anchor"] = None
        for s in (s for s in self.squares if s.kind == "number"):
            near = [line for line in lines if line["anchor"] is None and s.box[2] - 0.5 * unit <= line["box"][0] <= s.box[2] + 1.5 * unit
                    and geo.y_overlap(line["words"][0]["box"], s.box) >= 0.3 * min(s.box[3] - s.box[1], line["box"][3] - line["box"][1])]
            if near:
                min(near, key=lambda line: line["box"][0])["anchor"] = s
        tolerance, ratio = (1.15, 0.85) if self.work.source == "text_layer" else (1.3, 0.8)
        blocks: list[_Block] = []
        for line in sorted(lines, key=lambda line: (line["box"][1], line["box"][0])):
            box, cell, size = line["box"], self.cell_of(line["box"]), _Block([line]).size
            target = None
            if line["anchor"] is None:
                options = []
                for block in blocks:
                    last = block.lines[-1]["box"]
                    gap = box[1] - last[3]
                    if not -0.3 * unit <= gap <= 0.8 * unit or block.cell != cell:
                        continue
                    if abs(box[0] - block.left) > 1.5 * unit and abs(geo.center(box)[0] - geo.center(block.box)[0]) > 1.5 * unit:
                        continue
                    if max(size, block.size) > tolerance * min(size, block.size) or self._rule_between(last, box):
                        continue
                    options.append((gap, block))
                target = min(options, key=lambda o: o[0])[1] if options else None
            if target is None:
                blocks.append(_Block([line], line["anchor"], cell))
            else:
                target.lines.append(line)
        for block in blocks:
            if block.anchor or not block.cell or block.role != "label":
                continue
            cell = block.cell
            if (cell[3] - cell[1]) - (block.box[3] - block.box[1]) < unit and geo.contained(block.box, cell) > 0.8:
                if any(other is not block and 0 <= cell[1] - other.box[3] <= 1.5 * unit and geo.x_overlap(other.box, cell) > 0
                       and other.size <= ratio * block.size for other in blocks):
                    block.role = "placeholder"
        if self.work.source == "text_layer":
            for block in blocks:
                if block.anchor or block.cell or block.role != "label":
                    continue
                above = [other for other in blocks if other is not block and other.role == "label" and 0 <= block.box[1] - other.box[3] <= 1.5 * unit
                         and geo.x_overlap(other.box, block.box) > 0 and other.size <= ratio * block.size]
                if above:
                    block.role = "placeholder"
                    block.owner = max(above, key=lambda other: other.box[3])
        for block in blocks:
            letters = sum(c.isalpha() for c in block.text)
            if block.role == "label" and not block.anchor and (letters == 0 or (letters == 1 and (block.confidence or 1) < 0.9) or (block.confidence or 1) < 0.5):
                block.role = "noise"
        return blocks

    def _main(self, blocks: list[_Block], limit: float | None = None) -> _Block | None:
        pool = [b for b in blocks if limit is None or b.box[1] <= limit]
        anchored = [b for b in pool if b.anchor]
        if anchored:
            return min(anchored, key=lambda b: (b.box[1], b.box[0]))
        return min(pool, key=lambda b: (round(b.box[1] / self.unit), b.box[0])) if pool else None

    def _subtract(self, region: Box, obstacles: list[Box], under: Box | None = None, depth: int = 6) -> Box | None:
        unit, pad = self.unit, 0.2 * self.unit
        x1, y1, x2, y2 = region
        if x2 - x1 < 2.5 * unit or y2 - y1 < 0.7 * unit or (under and geo.x_overlap(region, under) <= 0):
            return None
        hits = [o for o in obstacles if geo.intersection(o, region) > 0]
        if not hits:
            return region
        o = max(hits, key=geo.area)
        options = ((x1, y1, o[0] - pad, y2), (o[2] + pad, y1, x2, y2), (x1, y1, x2, o[1] - pad), (x1, o[3] + pad, x2, y2))
        if depth <= 0:
            return self._subtract(max(options, key=geo.area), hits, under, 0)
        found = [r for option in options if (r := self._subtract(option, hits, under, depth - 1))]
        return max(found, key=geo.area, default=None)

    def _below(self, block: _Block) -> Box | None:
        unit, box = self.unit, block.box
        height, width = self.h.shape
        top = int(box[3] + 0.5 * unit)
        rows = np.flatnonzero(self.h[top:min(height, top + int(8 * unit)), int(box[0]):int(box[2])].any(axis=1))
        bottom = top + rows[0] if rows.size else box[3] + 2 * unit
        band = self.v[int(box[1]):int(bottom), int(box[2]):min(width, int(self.form[2]))]
        cols = np.flatnonzero((band > 0).mean(axis=0) >= 0.5) if band.size else np.zeros(0)
        right = box[2] + cols[0] if cols.size else min(width, self.form[2] - unit)
        span = (box[2], box[1], right, bottom)
        stops = [b.left for b in self.blocks if b is not block and b.role != "noise" and b.box[0] > box[2] and geo.y_overlap(b.box, span) > 0]
        stops += [s.box[0] for s in self.squares if s.box[0] > box[2] and geo.y_overlap(s.box, span) > 0]
        stops += [e[0][0] for e in self.runs if e[0][0] > box[2] and geo.y_overlap(geo.union(e), span) > 0]
        right = min([right] + [x - 0.3 * unit for x in stops])
        region = (block.left, box[3] + 0.2 * unit, right - 0.2 * unit, bottom - 0.2 * unit)
        if any(b is not block and b.role != "noise" and geo.intersection(b.box, region) >= 0.3 * geo.area(b.box) for b in self.blocks):
            return None
        return region

    def _layout(self, words: list[dict]) -> list[_Draft]:
        unit = self.unit
        used = self._options(words)
        blocks = self._blocks([w for i, w in enumerate(words) if i not in used and (not w["weak"] or self._anchored(w["box"]))])
        self.blocks = blocks
        labels = [b for b in blocks if b.role == "label"]
        by_cell: dict[Box, list[_Block]] = defaultdict(list)
        for block in labels:
            if block.cell:
                by_cell[block.cell].append(block)
        drafts: list[_Draft] = []
        obstacles: dict[Box | None, list[Box]] = defaultdict(list)
        for elements in self.runs:
            box = geo.union(elements)
            cell = self.cell_of(box)
            block = self._run_label(box, cell, by_cell.get(cell, [])) if cell else None
            block = block or self._run_label(box, None, [b for b in labels if geo.distance(b.box, box) <= 3 * unit
                                                         or (b.anchor and 0 <= box[1] - b.box[3] <= 5 * unit and geo.x_overlap(b.box, box) > 0)])
            if block:
                block.used = True
            obstacles[cell].append(box)
            drafts.append(_Draft("digit_sequence", box, block, [{"index": i, "bbox": e} for i, e in enumerate(elements)]))
        for cell, squares, block in self._checkbox_groups(by_cell, labels):
            boxes = [s.box for s in squares] + [s.label_box for s in squares if s.label_box]
            region = geo.inset(geo.union(boxes), -0.2 * unit)
            if block:
                block.used = True
            obstacles[cell].append(region)
            ordered = sorted(squares, key=lambda s: (round(s.box[1] / unit), s.box[0]))
            drafts.append(_Draft("checkbox", region, block, [{"label": s.label, "code": s.code, "bbox": s.box} for s in ordered]))
        for cell in self.cells:
            blocks_in = by_cell.get(cell, [])
            if blocks_in:
                drafts += self._cell_fields(cell, blocks_in, obstacles[cell])
        filled = set(by_cell) | {self.cell_of(s.box) for s in self.squares if s.kind == "checkbox"} | {self.cell_of(geo.union(e)) for e in self.runs}
        filled |= {self.cell_of(words[i]["box"]) for i in used}
        drafts += self._empty_cells(labels, filled)
        for block in blocks:
            if block.role == "placeholder" and not block.cell and block.owner and not block.owner.used:
                block.owner.used = True
                drafts.append(_Draft("text", geo.inset(block.box, -0.2 * unit), block.owner))
        for block in labels:
            if block.anchor and not block.used and (region := self._below(block)):
                block.used = True
                drafts.append(_Draft("text", region, block))
        for block in labels:
            heads = [o for o in labels if o.anchor and abs(o.box[1] - block.box[1]) <= 0.5 * unit and o.box[2] + unit <= block.box[0]]
            if block.used or block.cell or block.anchor or not any(not self._anchor_between(o.box, block.box) for o in heads):
                continue
            region = self._below(block)
            if region and not any(geo.intersection(region, d.region) > 0.3 * geo.area(region) for d in drafts):
                block.used = True
                drafts.append(_Draft("text", region, block))
        return drafts

    def _run_label(self, box: Box, cell: Box | None, pool: list[_Block]) -> _Block | None:
        unit = self.unit
        adjacent = [b for b in pool if 0 <= box[0] - b.box[2] <= 1.5 * unit and geo.y_overlap(b.box, box) >= 0.3 * (b.box[3] - b.box[1])]
        if adjacent:
            return max(adjacent, key=lambda b: b.box[2])
        above = [b for b in pool if b.box[3] <= geo.center(box)[1] + 0.2 * unit and b.box[1] < box[1] and geo.x_overlap(b.box, box) > 0]
        if above:
            return max(above, key=lambda b: b.box[3])
        left = [b for b in pool if b.box[2] <= box[0] + 0.5 * unit and geo.y_overlap(b.box, box) > 0]
        if left:
            return max(left, key=lambda b: b.box[2])
        return self._main(pool, box[1]) if cell else None

    def _checkbox_groups(self, by_cell: dict, labels: list[_Block]) -> list[tuple[Box | None, list[_Square], _Block | None]]:
        unit = self.unit

        def extent(squares: list[_Square]) -> Box:
            return geo.union([s.box for s in squares] + [s.label_box for s in squares if s.label_box])

        grouped: dict[tuple, list[_Square]] = defaultdict(list)
        loose: list[list[_Square]] = []
        for s in (s for s in self.squares if s.kind == "checkbox"):
            cell = self.cell_of(s.box)
            if cell:
                owners = [b for b in by_cell.get(cell, []) if b.anchor and b.left <= s.box[0] + 0.5 * unit and b.box[1] <= s.box[1]]
                owner = max(owners, key=lambda b: (b.box[1] // (2 * unit), b.left)) if owners else None
                grouped[cell, owner].append(s)
                continue
            near = [g for g in loose if any(geo.distance(s.box, t.box) <= (8 if geo.y_overlap(s.box, t.box) > 0 else 2.5) * unit
                                            and not self._wall(s.box, t.box) and not self._anchor_between(s.box, t.box) for t in g)]
            merged = [s] + [t for g in near for t in g]
            loose = [g for g in loose if g not in near] + [merged]
        groups = [[cell, squares, owner or self._main(by_cell.get(cell, []), min(s.box[1] for s in squares) + unit)]
                  for (cell, owner), squares in grouped.items()]
        groups += [[None, squares, None] for squares in loose]

        def hosts(group: list, box: Box) -> list:
            return [g for g in groups if g is not group and g[2] is not None and g[1] and geo.y_overlap(extent(g[1]), box) > 0
                    and 0 <= box[0] - extent(g[1])[2] <= 10 * unit and not self._anchor_between(extent(g[1]), box)]

        for group in groups:
            if group[2] is None:
                box = extent(group[1])
                above = [b for b in labels if 0 <= box[1] - b.box[3] <= 3 * unit and geo.x_overlap(b.box, box) > 0 and (group[0] is None or b.cell != group[0])]
                anchored = [b for b in labels if b.anchor and b.box[3] <= box[1] + unit and box[1] - b.box[3] <= 4 * unit and b.left <= box[0]]
                if not hosts(group, box) and (above or anchored):
                    group[2] = max(above, key=lambda b: b.box[3]) if above else max(anchored, key=lambda b: b.left)
        for group in sorted(groups, key=lambda g: -extent(g[1])[0]):
            if group[2] is not None:
                continue
            found = hosts(group, extent(group[1]))
            if found:
                host = max(found, key=lambda g: extent(g[1])[2])
                host[1] += group[1]
                group[1] = []
        merged: dict[int, list] = {}
        for group in groups:
            if group[1] and group[2] is not None and id(group[2]) in merged:
                merged[id(group[2])][1] += group[1]
                group[1] = []
            elif group[1] and group[2] is not None:
                merged[id(group[2])] = group
        return [(cell, squares, block) for cell, squares, block in groups if squares]

    def _cell_fields(self, cell: Box, blocks: list[_Block], obstacles: list[Box]) -> list[_Draft]:
        unit = self.unit
        main = self._main(blocks)
        fields = [main] + [b for b in blocks if b is not main and (b.anchor or (abs(b.box[1] - main.box[1]) <= unit and b.box[0] >= main.box[2] + 0.5 * unit))]
        others = [b.box for b in blocks if b not in fields]
        drafts = []
        for block in (b for b in fields if not b.used):
            row = [b for b in fields if b is not block and abs(b.box[1] - block.box[1]) <= unit]
            left = block.left - 0.3 * unit if any(b.left < block.left for b in row) else cell[0] + 0.2 * unit
            right = min([b.left for b in row if b.left > block.left], default=cell[2] + 0.1 * unit) - 0.3 * unit
            below = [b.box[1] for b in fields if b.box[1] > block.box[3] and geo.x_overlap(b.box, (left, 0, right, 0)) > 0]
            region = (left, block.box[3] + 0.2 * unit, right, min(below, default=cell[3]) - 0.2 * unit)
            free = self._subtract(region, obstacles + [o for o in others if geo.intersection(o, region) > 0], block.box)
            if free is None and block.anchor:
                free = region if region[3] - region[1] >= 0.5 * unit else self._below(block)
            if free is not None:
                block.used = True
                drafts.append(_Draft("text", free, block))
        return drafts

    def _empty_cells(self, labels: list[_Block], filled: set) -> list[_Draft]:
        unit, pairs, drafts = self.unit, [], []
        empty = [c for c in self.cells if c not in filled]
        for cell in empty:
            for block in labels:
                gap = cell[1] - block.box[3]
                if not block.used and -0.3 * unit <= gap <= 2 * unit and geo.x_overlap(block.box, cell) >= 0.5 * (block.box[2] - block.box[0]):
                    pairs.append((gap, abs(block.box[0] - cell[0]), cell, block))
        taken: set[Box] = set()
        for _, _, cell, block in sorted(pairs, key=lambda p: p[:2]):
            if cell in taken or block.used:
                continue
            taken.add(cell)
            block.used = True
            drafts.append(_Draft("text", cell, block))
        for cell in empty:
            if cell in taken:
                continue
            anchor = min((s for s in self.squares if s.kind == "number" and self.cell_of(s.box) == cell), key=lambda s: s.box[1], default=None)
            region = geo.inset(cell, 0.1 * unit)
            if anchor:
                region = (region[0], anchor.box[3] + 0.2 * unit, region[2], region[3])
            height, width = region[3] - region[1], region[2] - region[0]
            if width >= 1.5 * unit and unit <= height and (height <= 4 * unit or anchor or not self.blocks):
                drafts.append(_Draft("text", region, anchor=anchor))
        return drafts


def _neighbors(draft: _Draft, blocks: list[_Block], unit: float) -> tuple[str, ...]:
    anchor = draft.block.box if draft.block else draft.region
    own = draft.block.text if draft.block else None
    scored = []
    for block in blocks:
        if block is draft.block or block.role != "label" or not block.text or block.text == own:
            continue
        box = block.box
        above = box[3] <= anchor[1] + 0.5 * unit
        left = box[2] <= anchor[0] + 0.5 * unit and geo.y_overlap(box, anchor) > -unit
        if (above or left or geo.y_overlap(box, anchor) > 0) and geo.distance(box, anchor) <= 30 * unit:
            scored.append((geo.distance(box, anchor), block.text))
    out: list[str] = []
    for _, text in sorted(scored):
        if text not in out:
            out.append(text)
    return tuple(out[:4])


def _order(drafts: list[_Draft], unit: float) -> list[_Draft]:
    def anchor(d: _Draft) -> Box:
        return d.block.box if d.block else d.region
    ordered, band, top = [], [], None
    for draft in sorted(drafts, key=lambda d: anchor(d)[1]):
        if top is not None and anchor(draft)[1] - top > unit:
            ordered += sorted(band, key=lambda d: anchor(d)[0])
            band, top = [], None
        band.append(draft)
        top = anchor(draft)[1] if top is None else top
    return ordered + sorted(band, key=lambda d: anchor(d)[0])


def analyze_document(pdf: Path | bytes, page: int = 0, ocr: bool = True) -> PageAnalysis:
    started = time.perf_counter()
    timings: dict[str, float] = {}
    warnings: list[str] = []
    with _open(pdf) as doc:
        pages = doc.page_count
        if not 0 <= page < pages:
            raise ValueError(f"Página {page + 1} inexistente: o PDF tem {pages} página(s).")
        pg = doc[page]
        size = (math.ceil(round(pg.rect.width * SCALE, 6)), math.ceil(round(pg.rect.height * SCALE, 6)))
        if pages > 1:
            warnings.append(f"O PDF tem {pages} páginas; apenas a página {page + 1} foi analisada.")
        work = _load(doc, pg, size, warnings)
    timings["load"] = round(time.perf_counter() - started, 3)
    analyzer = _Analyzer(work, ocr, warnings, timings)
    drafts, unit = analyzer.run()
    ox, oy, sx, sy = work.transform

    def box(b: Box) -> tuple[int, int, int, int]:
        x1, x2 = sorted(min(max(round(ox + v * sx), 0), size[0] - 1) for v in (b[0], b[2]))
        y1, y2 = sorted(min(max(round(oy + v * sy), 0), size[1] - 1) for v in (b[1], b[3]))
        return x1, y1, x2, y2

    titles = analyzer.titles if work.source == "raster" else [" ".join(w["text"] for w in sorted(words, key=lambda w: -w["box"][1]))
                                                              for words in analyzer.bar_words]
    candidates = []
    if work.source == "text_layer":
        source = "text_layer"
    else:
        source = "tesseract" if analyzer.ocr_ok else "geometry"
    for index, draft in enumerate(_order(drafts, unit), 1):
        cy = geo.center(draft.region)[1]
        section = None
        if analyzer.bars:
            gaps = [max(0.0, b[1] - cy, cy - b[3]) for b in analyzer.bars]
            section = min(range(len(gaps)), key=gaps.__getitem__) if min(gaps) <= 3 * unit else None
        block = draft.block
        elements = tuple({**e, "bbox": list(box(e["bbox"]))} for e in draft.elements)
        candidates.append(FieldCandidate(
            id=f"c{index:03d}", label=block.text if block else "", label_bbox=box(block.box) if block else None, region=box(draft.region),
            kind=draft.kind, elements=elements, section=roman(section + 1) if section is not None else None,
            section_title=(titles[section] or None) if section is not None else None,
            neighbors=_neighbors(draft, analyzer.blocks, unit), number=block.anchor.number if block and block.anchor else draft.anchor.number if draft.anchor else None,
            source=source if block else "geometry", confidence=block.confidence if block else None))
    if not candidates:
        warnings.append("Nenhum campo detectado na página.")
    timings["total"] = round(time.perf_counter() - started, 3)
    return PageAnalysis(page=page, dpi=DPI, page_size_px=size, unit=round(unit * (sx + sy) / 2, 2), source=source,
                        candidates=tuple(candidates), warnings=tuple(warnings), timings=timings)
