from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

Box = tuple[float, float, float, float]


def kernel(width: float, height: float) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_RECT, (max(1, int(round(width))), max(1, int(round(height)))))


def binarize(gray: np.ndarray, threshold: float) -> np.ndarray:
    return np.where(gray < threshold, 255, 0).astype(np.uint8)


def area(box: Sequence[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def intersection(a: Sequence[float], b: Sequence[float]) -> float:
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    shared = intersection(a, b)
    total = area(a) + area(b) - shared
    return shared / total if total > 0 else 0.0


def contained(inner: Sequence[float], outer: Sequence[float]) -> float:
    size = area(inner)
    return intersection(inner, outer) / size if size > 0 else 0.0


def x_overlap(a: Sequence[float], b: Sequence[float]) -> float:
    return min(a[2], b[2]) - max(a[0], b[0])


def y_overlap(a: Sequence[float], b: Sequence[float]) -> float:
    return min(a[3], b[3]) - max(a[1], b[1])


def center(box: Sequence[float]) -> tuple[float, float]:
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def holds(box: Sequence[float], point: tuple[float, float]) -> bool:
    return box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]


def union(boxes: Sequence[Sequence[float]]) -> Box:
    return min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)


def inset(box: Sequence[float], dx: float, dy: float | None = None) -> Box:
    dy = dx if dy is None else dy
    return box[0] + dx, box[1] + dy, box[2] - dx, box[3] - dy


def distance(a: Sequence[float], b: Sequence[float]) -> float:
    dx = max(0.0, a[0] - b[2], b[0] - a[2])
    dy = max(0.0, a[1] - b[3], b[1] - a[3])
    return float(np.hypot(dx, dy))


def ink(mask: np.ndarray, box: Sequence[float], keep: float = 1.0) -> float:
    x1, y1, x2, y2 = box
    mx, my = (x2 - x1) * (1 - keep) / 2, (y2 - y1) * (1 - keep) / 2
    crop = mask[max(0, int(y1 + my)):max(0, int(np.ceil(y2 - my))), max(0, int(x1 + mx)):max(0, int(np.ceil(x2 - mx)))]
    return float((crop > 0).mean()) if crop.size else 0.0


def ring_ink(mask: np.ndarray, box: Sequence[float], near: float, far: float) -> float:
    height, width = mask.shape
    outer = [int(max(0, box[0] - far)), int(max(0, box[1] - far)), int(min(width, box[2] + far)), int(min(height, box[3] + far))]
    crop = mask[outer[1]:outer[3], outer[0]:outer[2]] > 0
    if not crop.size:
        return 0.0
    keep = np.ones(crop.shape, bool)
    a, b = int(box[1] - near) - outer[1], int(box[0] - near) - outer[0]
    keep[max(0, a):max(0, int(box[3] + near) - outer[1]), max(0, b):max(0, int(box[2] + near) - outer[0])] = False
    return float(crop[keep].mean()) if keep.any() else 0.0


def frame(mask: np.ndarray, box: Sequence[float], depth: int = 2) -> float:
    x1, y1, x2, y2 = (int(v) for v in box)
    h, w = mask.shape
    mx, my = max(1, (x2 - x1) // 10), max(1, (y2 - y1) // 10)
    sides = [mask[max(0, y1 - depth + 1):y1 + 1, x1 + mx:x2 - mx].any(axis=0), mask[y2 - 1:min(h, y2 + depth - 1), x1 + mx:x2 - mx].any(axis=0),
             mask[y1 + my:y2 - my, max(0, x1 - depth + 1):x1 + 1].any(axis=1), mask[y1 + my:y2 - my, x2 - 1:min(w, x2 + depth - 1)].any(axis=1)]
    corners = [mask[max(0, y - 1):y + 2, max(0, x - 1):x + 2].any() for x in (x1, x2 - 1) for y in (y1, y2 - 1)]
    return min((float(side.mean()) if side.size else 0.0) for side in sides) if all(corners) else 0.0


def line_masks(bw: np.ndarray, length: float, bridge: float, hbridge: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    length = max(9, int(length))
    horizontal = cv2.morphologyEx(cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel(length, 1)), cv2.MORPH_CLOSE, kernel(max(3, hbridge or bridge), 1))
    vertical = cv2.morphologyEx(cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel(1, length)), cv2.MORPH_CLOSE, kernel(1, max(3, bridge)))
    return horizontal, vertical


def clear(crop: np.ndarray, limit: float = 0.01) -> bool:
    return not crop.size or float((crop > 0).mean()) < limit


def complete_lines(horizontal: np.ndarray, vertical: np.ndarray, vmin: float, maxgap: float, hgap: float, band: int,
                   bw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h2, v2 = horizontal.copy(), vertical.copy()
    height, width = horizontal.shape
    gap = int(maxgap)
    rows = horizontal > 0
    _, _, stats, _ = cv2.connectedComponentsWithStats(vertical, connectivity=8)
    for x, y, w, h, _ in stats[1:]:
        if h < vmin:
            continue
        xc = x + w // 2
        hits = rows[:, max(0, xc - band):xc + band + 1].any(axis=1)
        if not hits[max(0, y - band):y + band + 1].any():
            start = max(0, y - gap)
            up = np.flatnonzero(hits[start:y])
            if up.size and clear(bw[start + up[-1] + band:y - 1, x:x + w]):
                v2[start + up[-1]:y + 1, x:x + w] = 255
        end = y + h
        if not hits[max(0, end - band - 1):end + band].any():
            down = np.flatnonzero(hits[end:min(height, end + gap)])
            if down.size and clear(bw[end + 1:end + down[0] - band, x:x + w]):
                v2[end - 1:end + down[0] + 1, x:x + w] = 255
    columns, gap = v2 > 0, int(hgap)
    _, _, stats, _ = cv2.connectedComponentsWithStats(horizontal, connectivity=8)
    for x, y, w, h, _ in stats[1:]:
        yc = y + h // 2
        hits = columns[max(0, yc - band):yc + band + 1, :].any(axis=0)
        if not hits[max(0, x - band):x + band + 1].any():
            left = np.flatnonzero(hits[max(0, x - gap):x])
            if left.size:
                h2[y:y + h, max(0, x - gap) + left[-1]:x + 1] = 255
        end = x + w
        if not hits[max(0, end - band - 1):end + band].any():
            right = np.flatnonzero(hits[end:min(width, end + gap)])
            if right.size:
                h2[y:y + h, end - 1:end + right[0] + 1] = 255
    return h2, v2


def cells(horizontal: np.ndarray, vertical: np.ndarray, grow: int, min_size: float, min_fill: float = 0.6) -> list[Box]:
    grow = max(3, int(grow))
    grid = cv2.dilate(cv2.bitwise_or(horizontal, vertical), kernel(grow, grow))
    _, _, stats, _ = cv2.connectedComponentsWithStats(cv2.bitwise_not(grid), connectivity=4)
    height, width = grid.shape
    out = []
    for x, y, w, h, size in stats[1:]:
        if w < min_size or h < min_size or x <= 0 or y <= 0 or x + w >= width or y + h >= height:
            continue
        if size >= min_fill * w * h:
            out.append((float(x - grow // 2), float(y - grow // 2), float(x + w + grow // 2), float(y + h + grow // 2)))
    return out


def rect_holes(bw: np.ndarray, low: float, high: float, rect_min: float = 0.8) -> list[Box]:
    contours, hierarchy = cv2.findContours(bw, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return []
    out = []
    for contour, (_, _, _, parent) in zip(contours, hierarchy[0]):
        if parent == -1:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if low <= w <= high and low <= h <= high and cv2.contourArea(contour) >= rect_min * w * h:
            out.append((float(x), float(y), float(x + w), float(y + h)))
    return out


def strokes(bw: np.ndarray, horizontal: np.ndarray, length: float, reach: float, max_width: float) -> list[dict]:
    opened = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel(1, max(5, int(length))))
    _, _, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
    near = cv2.dilate(horizontal, kernel(3, 3))
    reach = max(2, int(reach))
    out = []
    for x, y, w, h, _ in stats[1:]:
        if w > max_width:
            continue
        bottom = y + h
        touch = bool(near[max(0, bottom - reach):min(near.shape[0], bottom + reach), x:x + w].any())
        xc = int(x + w / 2)
        above = np.flatnonzero(horizontal[:y + reach, max(0, xc - 2):xc + 3].any(axis=1))
        above = above[above < bottom - reach]
        row = bottom - above[-1] if above.size else bottom
        out.append({"x": x + w / 2, "y2": float(bottom), "len": float(h), "touch": touch, "ratio": h / max(1, row)})
    return out


def digit_runs(found: list[dict], ytol: float, min_pitch: float, max_pitch: float, tick_max: float, cv_max: float = 0.2,
               ratio_max: float = 0.6) -> list[dict]:
    groups: list[list[dict]] = []
    for stroke in sorted((s for s in found if s["touch"]), key=lambda s: s["y2"]):
        for group in groups:
            if abs(group[0]["y2"] - stroke["y2"]) <= ytol:
                group.append(stroke)
                break
        else:
            groups.append([stroke])
    runs = []

    def flush(current: list[dict]) -> None:
        if len(current) < 3:
            return
        steps = np.diff([s["x"] for s in current])
        if steps.min() < min_pitch or steps.std() / steps.mean() > cv_max:
            return
        inner = [s for s in current[1:-1] if not s.get("virtual")]
        short = sum(s["len"] <= tick_max and s["ratio"] <= ratio_max for s in inner)
        if not inner or len(inner) < (len(current) - 2) / 2 or short < 0.75 * len(inner):
            return
        runs.append({"xs": [s["x"] for s in current], "baseline": float(np.median([s["y2"] for s in current])),
                     "pitch": float(steps.mean())})

    for group in groups:
        ordered: list[dict] = []
        for stroke in sorted(group, key=lambda s: s["x"]):
            if ordered and stroke["x"] - ordered[-1]["x"] < min_pitch * 0.5:
                if stroke["len"] > ordered[-1]["len"]:
                    ordered[-1] = stroke
                continue
            ordered.append(stroke)
        xs = [s["x"] for s in ordered]
        steps = np.diff(xs)
        extra = []
        for i, step in enumerate(steps):
            near = [steps[j] for j in (i - 2, i - 1, i + 1, i + 2) if 0 <= j < len(steps) and min_pitch <= steps[j] <= max_pitch]
            if len(near) < 2:
                continue
            ref = float(np.median(near))
            k = round(step / ref)
            if k in (2, 3) and abs(step - k * ref) <= 0.12 * k * ref:
                extra += [{**ordered[i], "x": xs[i] + j * step / k, "virtual": True} for j in range(1, k)]
        ordered = sorted(ordered + extra, key=lambda s: s["x"])
        current = ordered[:1]
        for stroke in ordered[1:]:
            step = stroke["x"] - current[-1]["x"]
            last = current[-1]["x"] - current[-2]["x"] if len(current) >= 2 else step
            if step > max_pitch or (len(current) >= 2 and abs(step - last) > 0.35 * last):
                flush(current)
                current = [current[-1], stroke] if step <= max_pitch else [stroke]
            else:
                current.append(stroke)
        flush(current)
    return runs


def comb_runs(bw: np.ndarray, horizontal: np.ndarray, dpi: float, unit: float) -> list[dict]:
    found = []
    for length in (max(0.06 * dpi, 0.6 * unit), max(0.035 * dpi, 0.4 * unit)):
        found += digit_runs(strokes(bw, horizontal, length, min(0.03 * dpi, 0.3 * unit), 0.055 * dpi), min(0.04 * dpi, 0.5 * unit), min(0.12 * dpi, 2 * unit),
                            min(0.75 * dpi, 4.5 * unit), 0.3 * dpi)
    found.sort(key=lambda r: -len(r["xs"]))
    keep: list[dict] = []
    for run in found:
        if not any(abs(run["baseline"] - k["baseline"]) < 0.05 * dpi and min(run["xs"][-1], k["xs"][-1]) > max(run["xs"][0], k["xs"][0])
                   for k in keep):
            keep.append(run)
    return keep


def run_elements(run: dict, dpi: float, height_factor: float = 0.8) -> list[Box]:
    pad = 0.02 * dpi
    return [(a + pad, run["baseline"] - height_factor * (b - a), b - pad, run["baseline"]) for a, b in zip(run["xs"], run["xs"][1:])]


def bars(gray: np.ndarray, unit: float, x_limit: float) -> list[Box]:
    dark = cv2.morphologyEx(binarize(gray, 110), cv2.MORPH_CLOSE, kernel(0.5 * unit, 0.5 * unit))
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, kernel(1, 2 * unit))
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, kernel(0.4 * unit, 1))
    _, _, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    out = []
    for x, y, w, h, size in stats[1:]:
        if x + w > x_limit or not 1.2 * unit <= w <= 8 * unit or h < 2 * unit or h < 1.1 * w or size < 0.5 * w * h:
            continue
        rows = (dark[y:y + h, x:x + w] > 0).mean(axis=1) >= 0.3
        edges = np.flatnonzero(np.diff(np.concatenate([[0], rows.astype(np.int8), [0]])))
        for top, bottom in zip(edges[::2], edges[1::2]):
            if bottom - top >= 2 * unit:
                out.append((float(x), float(y + top), float(x + w), float(y + bottom)))
    if out:
        left, right = float(np.median([b[0] for b in out])), float(np.median([b[2] for b in out]))
        out = [b for b in out if abs(b[0] - left) <= unit and abs(b[2] - right) <= 1.5 * unit]
    return sorted(out, key=lambda b: b[1])
