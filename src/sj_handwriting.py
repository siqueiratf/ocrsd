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

    def _lstm(self, inputs, state, index):
        torch, w = self.torch, self.weights
        suffix = "" if index == 0 else f"_{index}"
        prefix = f"rnn/LSTMAttentionCell/lstm_cell{suffix}"
        h, c = state
        i, j, f, o = (torch.cat((inputs, h), 1) @ w[prefix + "/kernel"]
                       + w[prefix + "/bias"]).chunk(4, 1)
        c = c * torch.sigmoid(f + 1) + torch.sigmoid(i) * torch.tanh(j)
        return torch.tanh(c) * torch.sigmoid(o), c

    def _step(self, inputs, states, context, kappa, embedding):
        torch, w = self.torch, self.weights
        states[0] = self._lstm(torch.cat((context, inputs), 1), states[0], 0)
        h0 = states[0][0]
        prefix = "rnn/LSTMAttentionCell/attention"
        params = torch.nn.functional.softplus(
            torch.cat((context, inputs, h0), 1) @ w[prefix + "/weights"] + w[prefix + "/biases"],
        )
        alpha, beta, delta = params.chunk(3, 1)
        kappa = kappa + delta / 25
        ruler = torch.arange(embedding.shape[0], dtype=torch.float32)
        phi = (alpha.T * torch.exp(-((kappa.T - ruler) ** 2) / beta.T.clamp(min=0.01))).sum(0)
        context = (phi @ embedding).unsqueeze(0)
        states[1] = self._lstm(torch.cat((inputs, h0, context), 1), states[1], 1)
        states[2] = self._lstm(torch.cat((inputs, states[1][0], context), 1), states[2], 2)
        return states, context, kappa, phi

    def _output(self, state, bias, rng):
        torch, w = self.torch, self.weights
        params = (state @ w["rnn/gmm/weights"] + w["rnn/gmm/biases"])[0]
        probabilities = torch.softmax(params[:20] * (1 + bias), 0).numpy().astype(float)
        probabilities[probabilities < 0.01] = 0
        probabilities /= probabilities.sum()
        index = rng.choice(20, p=probabilities)
        sigmas = np.maximum(np.exp(params[20:60].numpy() - bias), 1e-4)
        rho = float(torch.tanh(params[60 + index]).clamp(-1 + 1e-8, 1 - 1e-8))
        z1, z2 = rng.normal(size=2)
        x = float(params[80 + index]) + sigmas[index] * z1
        y = float(params[100 + index]) + sigmas[20 + index] * (rho * z1 + math.sqrt(max(0, 1-rho*rho)) * z2)
        probability = float(torch.sigmoid(params[120]))
        pen_up = float(rng.random() < (probability if probability >= 0.01 else 0))
        return torch.tensor([[x, y, pen_up]], dtype=torch.float32)

    def sample(self, text: str, writer: int, bias: float, seed: int) -> np.ndarray:
        from src.neural_handwriting import NeuralPen
        from scipy.signal import savgol_filter

        text = prepare_text(text)
        NeuralPen(model="sjvasquez", writer=writer, bias=bias)
        if len(text) > 75 or not 0 <= writer <= 12:
            raise ValueError("sjvasquez supports lines up to 75 characters and styles 0 to 12")
        torch = self.torch
        prefix = np.load(self.root / f"styles/style-{writer}-chars.npy", allow_pickle=False).tobytes().decode("utf-8")
        prime = np.load(self.root / f"styles/style-{writer}-strokes.npy", allow_pickle=False)
        codes = [ALPHABET.index(c) for c in prefix + " " + text] + [0]

        embedding = torch.nn.functional.one_hot(torch.tensor(codes + [0] * 8), len(ALPHABET)).float()
        embedding[len(codes):] = 0
        rng = np.random.default_rng(seed)
        with torch.inference_mode():
            states = [(torch.zeros(1, 400), torch.zeros(1, 400)) for _ in range(3)]
            context, kappa = torch.zeros(1, 73), torch.zeros(1, 10)
            for point in prime:
                states, context, kappa, _ = self._step(
                    torch.tensor(point.reshape(1, 3), dtype=torch.float32), states, context, kappa, embedding,
                )
            inputs = self._output(states[2][0], bias, rng)
            offsets = []
            for _ in range(max(2000, len(text) * 100)):
                states, context, kappa, phi = self._step(inputs, states, context, kappa, embedding)
                stop = self._output(states[2][0], bias, rng)
                position = int(phi.argmax())
                if position >= len(codes) or (position >= len(codes) - 1 and stop[0, 2]):
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
                inputs = self._output(states[2][0], bias, rng)
                offsets.append(inputs[0].numpy().copy())
        raise ValueError(f"Model did not finish writing {text!r}; try another seed or style")
