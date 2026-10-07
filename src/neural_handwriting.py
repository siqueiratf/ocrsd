from __future__ import annotations

import hashlib
import math
import random
import string
import unicodedata
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

MODEL_ID = "aaron1259/handwriting-synthesis"
WPVN_MODEL_ID = "wpvnteam/handwriting"
REVISION = "7b79d5859188502b0aed5b56fc868808f4ae2612"
WPVN_REVISION = "b8bd0b0"
SHA256 = "f7fc4afc0434dc35045bb65fec7b9640541067a4969cfcce917fdd6cf59773d8"
WEIGHTS_PATH = Path(__file__).resolve().parent.parent / "assets/handwriting/weights.bin"
ALPHABET = '\x00 !"#\'(),-.0123456789:;?ABCDEFGHIJKLMNOPRSTUVWYabcdefghijklmnopqrstuvwxyz'
SHAPES = {
    "conv1_d__b": (73,), "conv1_d__w": (5, 73, 73),
    "embed__embeddings": (512, 73), "embed_1__embeddings": (73, 73),
    "linear__b": (31,), "linear__w": (512, 31),
    "lstm_attention_core/~/gru__b": (1536,),
    "lstm_attention_core/~/gru__w_h": (512, 1536),
    "lstm_attention_core/~/gru__w_i": (76, 1536),
    "lstm_attention_core/~/gru_1__b": (1536,),
    "lstm_attention_core/~/gru_1__w_h": (512, 1536),
    "lstm_attention_core/~/gru_1__w_i": (588, 1536),
    "lstm_attention_core/~/linear__b": (3,),
    "lstm_attention_core/~/linear__w": (588, 3),
}


@dataclass(frozen=True)
class NeuralPen:
    model: str = "sjvasquez"
    writer: int | None = None
    bias: float = 0.75
    color: str = "#193787"
    width: float = 1.5
    pressure: float = 0.25
    dropout: float = 0.0
    text_policy: str = "strict"

    def __post_init__(self):
        if self.model not in {"sjvasquez", "aaron1259", "wpvnteam"}:
            raise ValueError("Model must be sjvasquez, aaron1259 or wpvnteam")
        maximum = 12 if self.model == "sjvasquez" else 511
        if self.writer is not None and (not isinstance(self.writer, int) or not 0 <= self.writer <= maximum):
            raise ValueError(f"Writer must be between 0 and {maximum} for {self.model}")
        if not math.isfinite(self.bias) or not 0 <= self.bias <= 3:
            raise ValueError("Bias must be between 0 and 3")
        if not math.isfinite(self.width) or not 0 < self.width <= 40:
            raise ValueError("Pen width must be greater than 0 and at most 40 pixels")
        if not 0 <= self.pressure <= 1 or not 0 <= self.dropout <= 1:
            raise ValueError("Pressure and dropout must be between 0 and 1")
        if self.text_policy not in {"strict", "normalize"}:
            raise ValueError("Text policy must be strict or normalize")
        if len(self.color) != 7 or not self.color.startswith("#") or any(c not in string.hexdigits for c in self.color[1:]):
            raise ValueError("Pen color must be #RRGGBB")
        try:
            int(self.color[1:], 16)
        except ValueError as exc:
            raise ValueError("Pen color must be #RRGGBB") from exc

    def metadata(self) -> dict:
        if self.model == "sjvasquez":
            from src.sj_handwriting import MODEL_ID as model_id, REVISION as revision, CHECKPOINT_BLOB
            source = {"repository": model_id, "revision": revision, "checkpoint_git_blob": CHECKPOINT_BLOB}
        elif self.model == "wpvnteam":
            source = {"repository": WPVN_MODEL_ID, "revision": WPVN_REVISION, "weights_sha256": SHA256}
        else:
            source = {"repository": MODEL_ID, "revision": REVISION, "weights_sha256": SHA256}
        return {**source, **asdict(self)}


def load_model(name: str):
    if name == "sjvasquez":
        from src.sj_handwriting import SjHandwritingModel
        return SjHandwritingModel()
    if name in {"aaron1259", "wpvnteam"}:
        return HandwritingModel()
    raise ValueError(f"Unknown handwriting model: {name}")


def prepare_text(text: str, policy: str = "strict") -> str:
    if policy not in {"strict", "normalize"}:
        raise ValueError("Text policy must be strict or normalize")
    if policy == "normalize":
        text = "".join(c for c in unicodedata.normalize("NFD", text) if not unicodedata.combining(c))
        text = text.translate(str.maketrans({"Q": "q", "X": "x", "Z": "z", "/": "-"}))
    unsupported = sorted(set(text) - set(ALPHABET[1:]))
    if unsupported:
        raise ValueError(f"Model cannot write {unsupported!r}; use --text-policy normalize or the font backend")
    if not text.strip() or len(text) > 160:
        raise ValueError("Neural handwriting requires 1 to 160 characters with nonblank text")
    return text


def download_weights(model: str = "aaron1259", path: Path = WEIGHTS_PATH) -> Path:
    if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == SHA256:
        return path
    if model not in {"aaron1259", "wpvnteam"}:
        raise ValueError(f"Unknown handwriting weights source: {model}")
    path.parent.mkdir(parents=True, exist_ok=True)
    model_id = WPVN_MODEL_ID if model == "wpvnteam" else MODEL_ID
    revision = WPVN_REVISION if model == "wpvnteam" else REVISION
    url = f"https://huggingface.co/spaces/{model_id}/resolve/{revision}/weights.bin"
    with urllib.request.urlopen(url, timeout=60) as response:
        data = response.read()
    if hashlib.sha256(data).hexdigest() != SHA256:
        raise ValueError("Downloaded handwriting weights failed SHA-256 verification")
    path.write_bytes(data)
    return path


class HandwritingModel:
    def __init__(self, path: Path = WEIGHTS_PATH):
        import torch

        if not path.is_file():
            raise FileNotFoundError(
                "GRU handwriting weights missing; run: "
                "uv run ocrsd handwriting --model aaron1259 --download "
                "or uv run ocrsd handwriting --model wpvnteam --download"
            )
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != SHA256:
            raise ValueError("Handwriting weights failed SHA-256 verification")
        self.torch = torch
        flat = np.frombuffer(data, dtype="<f4")
        offset = 0
        self.weights = {}
        for name, shape in SHAPES.items():
            size = math.prod(shape)
            self.weights[name] = torch.from_numpy(flat[offset:offset + size].reshape(shape).copy())
            offset += size
        if offset != len(flat):
            raise ValueError("Unexpected handwriting tensor layout")

    def _gru(self, inputs, state, name):
        torch = self.torch
        prefix = f"lstm_attention_core/~/{name}__"
        w = self.weights
        projected = inputs @ w[prefix + "w_i"]
        z, r = torch.sigmoid(projected[:, :1024] + state @ w[prefix + "w_h"][:, :1024]
                             + w[prefix + "b"][:1024]).chunk(2, dim=1)
        candidate = torch.tanh(projected[:, 1024:] + (r * state) @ w[prefix + "w_h"][:, 1024:]
                               + w[prefix + "b"][1024:])
        return (1 - z) * state + z * candidate

    def sample(self, text: str, writer: int, bias: float, seed: int) -> np.ndarray:
        text = prepare_text(text)
        NeuralPen(model="aaron1259", writer=writer, bias=bias)
        torch, w = self.torch, self.weights
        rng = random.Random(seed)
        encoded = [ALPHABET.index(c) for c in text + "          " + text]
        with torch.inference_mode():
            embedding = w["embed_1__embeddings"][encoded].T.unsqueeze(0)
            embedding = torch.nn.functional.conv1d(
                embedding, w["conv1_d__w"].permute(2, 1, 0), w["conv1_d__b"], padding=2,
            ).squeeze(0).T + w["embed__embeddings"][writer]
            h0, h1 = torch.zeros(1, 512), torch.zeros(1, 512)
            context, location = torch.zeros(1, 73), torch.zeros(1, 1)
            inputs = torch.tensor([[0., 0., 1.]])
            ruler = torch.arange(len(encoded), dtype=torch.float32)
            points = [[0., 0., 1.]]
            for _ in range(max(2000, len(text) * 100)):
                h0 = self._gru(torch.cat((context, inputs), 1), h0, "gru")
                attention_input = torch.cat((context, inputs, h0), 1)
                attention = torch.nn.functional.softplus(
                    attention_input @ w["lstm_attention_core/~/linear__w"]
                    + w["lstm_attention_core/~/linear__b"],
                )
                alpha, beta, kappa = attention.chunk(3, 1)
                location = location + kappa / 25
                phi = alpha * torch.exp(-((location - ruler) ** 2) / beta)
                context = phi @ embedding
                h1 = self._gru(torch.cat((inputs, h0, context), 1), h1, "gru_1")
                params = (h1 @ w["linear__w"] + w["linear__b"])[0].numpy()
                component = int(np.argmax(params[25:30]))
                noise = []
                for _axis in range(2):
                    value = rng.gauss(0, 1)
                    while abs(value) > 5:
                        value = rng.gauss(0, 1)
                    noise.append(value)
                rx = noise[0] * math.exp(float(params[10 + component]) - bias)
                ry = noise[1] * math.exp(float(params[15 + component]) - bias)
                angle = -float(params[20 + component])
                dx = float(params[component]) + math.cos(angle) * rx - math.sin(angle) * ry
                dy = float(params[5 + component]) + math.sin(angle) * rx + math.cos(angle) * ry
                pen_up = float(rng.random() < float(torch.sigmoid(torch.tensor(params[30]))))
                inputs = torch.tensor([[dx, dy, pen_up]])
                points.append([points[-1][0] + dx * 3, points[-1][1] - dy * 3, pen_up])
                if float(location.item()) >= len(text) + 2:
                    result = np.asarray(points, dtype=np.float32)
                    if not np.isfinite(result).all():
                        raise ValueError("Model generated nonfinite pen coordinates")
                    return result
        raise ValueError(f"Model did not finish writing {text!r}; try another seed or writer")


def render_strokes(points: np.ndarray, width: int, height: int, pen: NeuralPen,
                   seed: int) -> Image.Image:
    if width < 4 or height < 4 or points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("Invalid image dimensions or pen trajectory")
    segments = [(a[:2], b[:2]) for a, b in zip(points[:-1], points[1:]) if not a[2]]
    if not segments:
        raise ValueError("Model generated no visible strokes")
    visible = np.asarray(segments)
    low, high = visible.min(axis=(0, 1)), visible.max(axis=(0, 1))
    padding = pen.width / 2 + 2
    available = np.array([width, height]) - 2 * padding
    if (available <= 0).any():
        raise ValueError("Field too small for the selected pen width")
    scale = float(np.min(available / np.maximum(high - low, 1)))
    offset = (np.array([width, height]) - (high - low) * scale) / 2
    supersampling = 4
    mask = Image.new("L", (width * supersampling, height * supersampling), 0)
    draw = ImageDraw.Draw(mask)
    rng = random.Random(seed)
    pressure, gap = 1., 0
    for start, end in segments:
        pressure = 0.85 * pressure + 0.15 * rng.uniform(1 - pen.pressure, 1)
        if gap == 0 and rng.random() < pen.dropout * 0.25:
            gap = rng.randint(1, 4)
        if gap:
            gap -= 1
            continue
        start, end = ((point - low) * scale + offset for point in (start, end))
        a, b = tuple(start * supersampling), tuple(end * supersampling)
        line_width = max(1, round(pen.width * supersampling * pressure))
        opacity = round(255 * pressure)
        draw.line((a, b), fill=opacity, width=line_width)
        radius = line_width / 2
        for x, y in (a, b):
            draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=opacity)
    mask = mask.resize((width, height), Image.Resampling.LANCZOS)
    color = tuple(int(pen.color[i:i+2], 16) for i in (1, 3, 5))
    image = Image.new("RGBA", (width, height), (*color, 0))
    image.putalpha(mask)
    return image
