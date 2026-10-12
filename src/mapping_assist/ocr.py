from __future__ import annotations

import csv
import io
import os
import re
import shutil
import subprocess
from typing import Callable, Sequence

import cv2
import numpy as np

from src.mapping_assist.geometry import union

LETTERS = "A-Za-zÀ-ÖØ-öø-ÿ"
_NUMBER_PREFIX = re.compile(rf"^(?:[\[|({{]?[0-9A-Za-z]{{0,3}}[\]|)}}]+(?=[{LETTERS}])|[\[|({{]*\d+[\]|)}}/]*(?=[A-ZÀ-Ö]))")
_EDGE_JUNK = re.compile(r"^[\[\]|_~`'‘’\"“”]+|[\[\]|_~`'‘’\"“”]+$")
_KEEP_SYMBOLS = {"/", "-", "–", "(", ")", "º", "ª", "&", "%", "+"}
_ALPHA = re.compile(rf"[{LETTERS}]")


def run_tesseract(image: np.ndarray, psm: int = 6, dpi: float = 300, options: Sequence[str] = (), timeout: float = 180) -> list[dict]:
    executable = shutil.which("tesseract")
    if executable is None:
        raise FileNotFoundError("tesseract")
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("falha ao codificar a imagem para o OCR")
    command = [executable, "stdin", "stdout", "-l", "por", "--psm", str(psm), "--dpi", str(int(round(dpi))), *options, "tsv"]
    result = subprocess.run(command, input=encoded.tobytes(), capture_output=True, timeout=timeout,
                            env={**os.environ, "OMP_THREAD_LIMIT": "1"})
    if result.returncode:
        raise RuntimeError(result.stderr.decode("utf-8", "replace").strip()[-300:] or f"código {result.returncode}")
    words = []
    for row in csv.DictReader(io.StringIO(result.stdout.decode("utf-8", "replace")), delimiter="\t", quoting=csv.QUOTE_NONE):
        text = (row.get("text") or "").strip()
        if row.get("level") != "5" or not text:
            continue
        x, y, w, h = (int(row[key]) for key in ("left", "top", "width", "height"))
        words.append({"text": text, "conf": float(row["conf"]), "box": (float(x), float(y), float(x + w), float(y + h))})
    return words


def clean_text(text: str) -> str:
    text = _EDGE_JUNK.sub("", _NUMBER_PREFIX.sub("", _EDGE_JUNK.sub("", text.strip())))
    if not text or (not any(c.isalnum() for c in text) and text not in _KEEP_SYMBOLS):
        return ""
    return text


def is_alpha(text: str) -> bool:
    return bool(_ALPHA.search(text))


def unit(words: Sequence[dict], factor: float = 1.0) -> float | None:
    heights = [w["box"][3] - w["box"][1] for w in words
               if len(re.findall(rf"[{LETTERS}]", w["text"])) >= 4 and w["text"].isalpha() and (w.get("conf") is None or w["conf"] >= 80)]
    if len(heights) < 5:
        heights = [w["box"][3] - w["box"][1] for w in words if is_alpha(w["text"]) and len(w["text"]) >= 3]
    return float(np.median(heights)) * factor if len(heights) >= 3 else None


def group_lines(words: Sequence[dict], unit_px: float, blocked: Callable[[tuple, tuple], bool], gap: float = 1.0) -> list[dict]:
    lines: list[dict] = []
    for word in sorted(words, key=lambda w: w["box"][0]):
        box = word["box"]
        height = box[3] - box[1]
        best = None
        for line in lines:
            last = line["words"][-1]["box"]
            shared = min(last[3], box[3]) - max(last[1], box[1])
            if shared < 0.5 * max(1.0, min(height, last[3] - last[1])):
                continue
            if -0.5 * unit_px < box[0] - last[2] <= gap * unit_px and not blocked(last, box):
                best = line
                break
        if best is None:
            lines.append({"words": [word]})
        else:
            best["words"].append(word)
    for line in lines:
        line["box"] = union([w["box"] for w in line["words"]])
        line["text"] = " ".join(w["text"] for w in line["words"])
    return lines
