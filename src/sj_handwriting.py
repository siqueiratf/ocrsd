from __future__ import annotations

import hashlib
import json
import math
import urllib.request
from pathlib import Path

import numpy as np

from src.neural_handwriting import ALPHABET, WEIGHTS_PATH, prepare_text

REVISION = "5f58984c3bc793bd8930bad7012cd2008d4a4a66"
MODEL_ID = "sjvasquez/handwriting-synthesis"
ROOT = WEIGHTS_PATH.parent / "sjvasquez"
CHECKPOINT_BLOB = "4f440c01e41cd71cfe2c9c37cde8cfdd06df8ac0"


def download_weights(root: Path = ROOT) -> Path:
    manifest = json.loads((WEIGHTS_PATH.parent / "sjvasquez-files.json").read_text())
    for relative, expected in manifest.items():
        path = root / relative
        data = path.read_bytes() if path.is_file() else b""
        def digest(value):
            return hashlib.sha1(f"blob {len(value)}\0".encode() + value).hexdigest()
        if digest(data) != expected:
            url = f"https://raw.githubusercontent.com/{MODEL_ID}/{REVISION}/{relative}"
            with urllib.request.urlopen(url, timeout=60) as response:
                data = response.read()
            if digest(data) != expected:
                raise ValueError(f"Checkpoint integrity check failed: {relative}")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
    target = root / "weights.npz"
    checksum = root / "weights.sha256"
    if (not target.is_file() or not checksum.is_file()
            or hashlib.sha256(target.read_bytes()).hexdigest() != checksum.read_text().strip()):
        try:
            import tensorflow as tf
        except ImportError as exc:
            raise ImportError("Checkpoint conversion needs: uv sync --all-extras") from exc
        reader = tf.train.load_checkpoint(str(root / "checkpoints/model-17900"))
        tensors = {
            name: reader.get_tensor(name) for name in reader.get_variable_to_shape_map()
            if name.startswith("rnn/") and not name.endswith(("/RMSProp", "/RMSProp_1"))
        }
        np.savez(target, **tensors)
        checksum.write_text(hashlib.sha256(target.read_bytes()).hexdigest() + "\n")
    return target


class SjHandwritingModel:
    def __init__(self, root: Path = ROOT):
        import torch

        path = root / "weights.npz"
        if not path.is_file():
            raise FileNotFoundError("Missing sjvasquez weights; run: uv run ocrsd handwriting --model sjvasquez --download")
        checksum = root / "weights.sha256"
        if not checksum.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != checksum.read_text().strip():
            raise ValueError("Converted weights failed integrity check; run: uv run ocrsd handwriting --model sjvasquez --download")
        self.root = root
        self.torch = torch
        with np.load(path, allow_pickle=False) as arrays:
            self.weights = {name: torch.from_numpy(arrays[name].copy()) for name in arrays.files}
        self._lstm_weights = [
            (self.weights[f"rnn/LSTMAttentionCell/lstm_cell{'' if i == 0 else f'_{i}'}/kernel"],
             self.weights[f"rnn/LSTMAttentionCell/lstm_cell{'' if i == 0 else f'_{i}'}/bias"])
            for i in range(3)
        ]
        self._attention_weights = self.weights["rnn/LSTMAttentionCell/attention/weights"]
        self._attention_biases = self.weights["rnn/LSTMAttentionCell/attention/biases"]
        self._gmm_weights = self.weights["rnn/gmm/weights"]
        self._gmm_biases = self.weights["rnn/gmm/biases"]
        # The prime state of each writer, up to the step where the text starts to get attention.
        # The prime uses no RNG. The strokes differ from a full prime only by float rounding.
        self._prime_cache: dict[int, tuple] = {}
        self._cache_primed_style = True
        self._prime_probe_pad = 80
        self._prime_split_threshold = 1e-6
        self._prime_split_margin = 10
        self._styles: dict[int, tuple] = {}
        self._max_batch = 256

    def _lstm(self, inputs, state, index):
        torch = self.torch
        kernel, bias = self._lstm_weights[index]
        h, c = state
        i, j, f, o = (torch.cat((inputs, h), 1) @ kernel + bias).chunk(4, 1)
        c = c * torch.sigmoid(f + 1) + torch.sigmoid(i) * torch.tanh(j)
        return torch.tanh(c) * torch.sigmoid(o), c

    def _step(self, inputs, states, context, kappa, embedding, ruler):
        torch = self.torch
        states[0] = self._lstm(torch.cat((context, inputs), 1), states[0], 0)
        h0 = states[0][0]
        params = torch.nn.functional.softplus(
            torch.cat((context, inputs, h0), 1) @ self._attention_weights + self._attention_biases,
        )
        alpha, beta, delta = params.chunk(3, 1)
        kappa = kappa + delta / 25
        phi = (alpha.unsqueeze(2) * torch.exp(-((kappa.unsqueeze(2) - ruler) ** 2) / beta.unsqueeze(2).clamp(min=0.01))).sum(1)
        context = (phi.unsqueeze(1) @ embedding).squeeze(1)
        states[1] = self._lstm(torch.cat((inputs, h0, context), 1), states[1], 1)
        states[2] = self._lstm(torch.cat((inputs, states[1][0], context), 1), states[2], 2)
        return states, context, kappa, phi

    def _probe_prime_split(self, prefix, prime):
        """Find the first prime step K where the attention on the text is more than a small
        threshold, minus a margin. Zero rows replace the text. Steps before K do not use the
        text, so all texts of this writer can use the state before step K."""
        torch = self.torch
        pad = self._prime_probe_pad
        text_start = len(prefix) + 1
        prefix_codes = [ALPHABET.index(c) for c in prefix + " "] + [0]
        total_len = len(prefix_codes) + pad
        embedding = torch.nn.functional.one_hot(
            torch.tensor(prefix_codes + [0] * pad), len(ALPHABET),
        ).float()
        embedding[len(prefix_codes):] = 0
        embedding = embedding.unsqueeze(0)
        ruler = torch.arange(total_len, dtype=torch.float32)
        states = [(torch.zeros(1, 400), torch.zeros(1, 400)) for _ in range(3)]
        context, kappa = torch.zeros(1, 73), torch.zeros(1, 10)
        snapshots = [(list(states), context, kappa)]
        split = len(prime)
        for n in range(1, len(prime) + 1):
            states, context, kappa, phi = self._step(
                torch.tensor(prime[n - 1].reshape(1, 3), dtype=torch.float32), states, context, kappa, embedding, ruler,
            )
            snapshots.append((list(states), context, kappa))
            if float(phi[0, text_start:].sum()) > self._prime_split_threshold:
                split = max(0, n - self._prime_split_margin)
                break
        cached_states, cached_context, cached_kappa = snapshots[split]
        return cached_states, cached_context, cached_kappa, split

    def _style(self, writer):
        if writer not in self._styles:
            prefix = np.load(self.root / f"styles/style-{writer}-chars.npy", allow_pickle=False).tobytes().decode("utf-8")
            prime = np.load(self.root / f"styles/style-{writer}-strokes.npy", allow_pickle=False)
            self._styles[writer] = prefix, prime
        return self._styles[writer]

    def check(self, text: str, writer: int, bias: float) -> str:
        """Return the prepared text, or raise ValueError when the model cannot write it with any seed."""
        from src.neural_handwriting import NeuralPen

        text = prepare_text(text)
        NeuralPen(model="sjvasquez", writer=writer, bias=bias)
        if len(text) > 75 or not 0 <= writer <= 12:
            raise ValueError("sjvasquez supports lines up to 75 characters and styles 0 to 12")
        return text

    def _row(self, text, writer, bias, seed):
        text = self.check(text, writer, bias)
        torch = self.torch
        prefix, prime = self._style(writer)
        codes = [ALPHABET.index(c) for c in prefix + " " + text] + [0]
        if self._cache_primed_style:
            if writer not in self._prime_cache:
                with torch.inference_mode():
                    self._prime_cache[writer] = self._probe_prime_split(prefix, prime)
            states, context, kappa, split = self._prime_cache[writer]
        else:
            states = [(torch.zeros(1, 400), torch.zeros(1, 400)) for _ in range(3)]
            context, kappa, split = torch.zeros(1, 73), torch.zeros(1, 10), 0
        return {"text": text, "codes": codes, "bias": float(bias), "rng": np.random.default_rng(seed),
                "prime": prime[split:].astype(np.float32), "states": states, "context": context, "kappa": kappa,
                "steps": 0, "offsets": [], "limit": max(2000, len(text) * 100)}

    @staticmethod
    def _points(offsets):
        from scipy.signal import savgol_filter

        if len(offsets) < 2:
            raise ValueError("Model produced no handwriting")
        points = np.array(offsets, dtype=np.float32)
        points[:, :2] = np.cumsum(points[:, :2], axis=0)
        for stroke in np.split(points, np.where(points[:, 2] == 1)[0] + 1):
            if len(stroke):
                stroke[:, :2] = savgol_filter(stroke[:, :2], 7, 3, axis=0, mode="nearest")
        design = np.column_stack((np.ones(len(points)), points[:, 0]))
        coefficients, _, rank, _ = np.linalg.lstsq(design, points[:, 1], rcond=None)
        if rank == 2:
            angle = math.atan(float(coefficients[1]))
            points[:, :2] = points[:, :2] @ np.array([
                [math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)],
            ])
        points[:, 1] *= -1
        if not np.isfinite(points).all():
            raise ValueError("Model produced nonfinite pen coordinates")
        return points

    def _mixture(self, state, rows):
        torch = self.torch
        params = state @ self._gmm_weights + self._gmm_biases
        scale = torch.tensor([[1 + row["bias"]] for row in rows], dtype=torch.float32)
        probabilities = torch.softmax(params[:, :20] * scale, 1).numpy().astype(float)
        probabilities[probabilities < 0.01] = 0
        probabilities /= probabilities.sum(1, keepdims=True)
        cdf = probabilities.cumsum(1)
        cdf /= cdf[:, -1:]
        rho = torch.tanh(params[:, 60:80]).clamp(-1 + 1e-8, 1 - 1e-8).numpy()
        pen = torch.sigmoid(params[:, 120]).numpy()
        params = params.numpy()
        bias = np.array([[row["bias"]] for row in rows], dtype=np.float32)
        return cdf, rho, np.where(pen >= 0.01, pen, 0), params, np.maximum(np.exp(params[:, 20:60] - bias), 1e-4)

    @staticmethod
    def _pick(mixture, lines, rows):
        """Draw a pen offset for each line. Each row uses its own RNG in a fixed order: one uniform
        for the mixture component (as rng.choice), two normals, and one uniform for the pen."""
        cdf, rho, pen, params, sigmas = (array[lines] for array in mixture)
        draws = np.array([(row["rng"].random(), *row["rng"].normal(size=2), row["rng"].random()) for row in rows])
        index = (cdf <= draws[:, :1]).sum(1)
        line = np.arange(len(lines))
        rho = rho[line, index].astype(float)
        z1, z2 = draws[:, 1], draws[:, 2]
        x = params[line, 80 + index].astype(float) + sigmas[line, index] * z1
        y = params[line, 100 + index].astype(float) + sigmas[line, 20 + index] * (
            rho * z1 + np.sqrt(np.maximum(0, 1 - rho * rho)) * z2)
        return np.column_stack((x, y, (draws[:, 3] < pen).astype(float))).astype(np.float32)

    def sample_many(self, requests) -> list:
        """Sample each (text, writer, bias, seed) request. The rows run in batches of up to
        max_batch rows, and a new row starts when a row stops. Each row uses its own RNG in the
        same order as one call of sample(). The strokes differ from a batch of one row only by
        float rounding. A row that fails gets its ValueError in place of the points."""
        torch = self.torch
        results: list = [None] * len(requests)
        waiting = []
        for index, request in enumerate(requests):
            try:
                waiting.append((index, self._row(*request)))
            except ValueError as exc:
                results[index] = exc
        if not waiting:
            return results
        length = max(len(row["codes"]) for _, row in waiting) + 8
        ruler = torch.arange(length, dtype=torch.float32)
        active, batch = [], None
        waiting.reverse()
        with torch.inference_mode():
            while waiting or active:
                if waiting and len(active) < self._max_batch:
                    added = []
                    while waiting and len(active) + len(added) < self._max_batch:
                        added.append(waiting.pop())
                    embedding = torch.zeros(len(added), length, len(ALPHABET))
                    for line, (_, row) in enumerate(added):
                        embedding[line, torch.arange(len(row["codes"])), torch.tensor(row["codes"])] = 1
                    parts = (
                        [torch.cat([row["states"][layer][part] for _, row in added]) for layer in range(3) for part in range(2)]
                        + [torch.cat([row["context"] for _, row in added]), torch.cat([row["kappa"] for _, row in added]), embedding]
                    )
                    batch = parts if batch is None else [torch.cat(pair) for pair in zip(batch, parts)]
                    active += added
                ready = [line for line, (_, row) in enumerate(active) if row["steps"] >= len(row["prime"])]
                done, going = set(), []
                if ready:
                    rows = [active[line][1] for line in ready]
                    mixture = self._mixture(batch[4][ready], rows)
                    for place, (row, point) in enumerate(zip(rows, self._pick(mixture, list(range(len(ready))), rows))):
                        index, line = active[ready[place]][0], ready[place]
                        if row["steps"] == len(row["prime"]):
                            row["inputs"] = point
                            continue
                        codes = len(row["codes"])
                        position = int(row["phi"][:codes + 8].argmax())
                        if position >= codes or (position >= codes - 1 and point[2]):
                            try:
                                results[index] = self._points(row["offsets"])
                            except ValueError as exc:
                                results[index] = exc
                            done.add(line)
                        elif row["steps"] - len(row["prime"]) >= row["limit"]:
                            results[index] = ValueError(f"Model did not finish writing {row['text']!r}; try another seed or style")
                            done.add(line)
                        else:
                            going.append(place)
                    if going:
                        rows = [rows[place] for place in going]
                        for row, point in zip(rows, self._pick(mixture, going, rows)):
                            row["inputs"] = point
                            row["offsets"].append(point)
                if done:
                    keep = [line for line in range(len(active)) if line not in done]
                    active = [active[line] for line in keep]
                    batch = [tensor[keep] for tensor in batch] if keep else None
                    if not active:
                        continue
                inputs = torch.from_numpy(np.stack([
                    row["prime"][row["steps"]] if row["steps"] < len(row["prime"]) else row["inputs"] for _, row in active
                ]))
                states = [(batch[0], batch[1]), (batch[2], batch[3]), (batch[4], batch[5])]
                states, context, kappa, phi = self._step(inputs, states, batch[6], batch[7], batch[8], ruler)
                batch = [*states[0], *states[1], *states[2], context, kappa, batch[8]]
                for line, (_, row) in enumerate(active):
                    row["steps"] += 1
                    row["phi"] = phi[line]
        return results

    def sample(self, text: str, writer: int, bias: float, seed: int) -> np.ndarray:
        result = self.sample_many([(text, writer, bias, seed)])[0]
        if isinstance(result, Exception):
            raise result
        return result
