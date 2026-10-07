# Local handwriting models

## sjvasquez

Source: https://github.com/sjvasquez/handwriting-synthesis
Revision: `5f58984c3bc793bd8930bad7012cd2008d4a4a66`.

`src/sj_handwriting.py` implements the equations in `rnn_cell.py`, style
priming in `demo.py`, and smoothing/baseline alignment in `drawing.py`.
The original TensorFlow checkpoint `model-17900` is converted to NumPy tensors
without retraining. TensorFlow is used only for checkpoint conversion;
inference uses PyTorch on CPU. The 13 original style examples are retained.
`sjvasquez-files.json` records the upstream Git blob hashes for the checkpoint
and style arrays. A separate SHA-256 checks the converted tensors on load.

Install: `uv run ocrsd handwriting --model sjvasquez --download`.
The conversion dependency is installed by `uv sync --all-extras` or
`uv sync --extra handwriting`. After conversion, TensorFlow is not imported
when generating handwriting. No license file is present in the upstream
repository at the pinned revision; this integration does not assign one.

## aaron1259

Source: https://huggingface.co/spaces/aaron1259/handwriting-synthesis
Revision: `7b79d5859188502b0aed5b56fc868808f4ae2612`.

`src/neural_handwriting.py` implements the inference equations from `script.js`
and the float32 tensor layout from `model_weights.js`, using PyTorch on CPU.
The weights are downloaded unchanged from the Space and verified with SHA-256.
They are not committed to this repository. No remote inference is used.

The source exposes 512 writer embeddings and a sampling bias from 0 to 3.
It does not include training code. Its alphabet lacks accented letters,
uppercase Q/X/Z, slash and other symbols. Strict mode rejects unsupported text.
Explicit normalization removes accents, lowercases Q/X/Z and replaces slash
with hyphen. Dataset metadata records the original and rendered values.
This does not give the model native Portuguese support.

The Space does not declare a license in its metadata or include a LICENSE
file at this revision. Model/source attribution is retained here; no license
for those upstream artifacts is inferred from this integration.

Install weights: `uv run ocrsd handwriting --model aaron1259 --download`.

## Shared rendering

Both models emit pen trajectories. A shared renderer draws the resulting
strokes with configurable RGB color, pixel width, pressure variation and
intermittent ink dropout. These effects are applied after neural synthesis.
The original font renderer remains available as `font`.

Both models share the same restricted alphabet. They do not natively support
Portuguese accents. `sjvasquez` accepts at most 75 characters per line;
`aaron1259` accepts at most 160 in this integration. Generation raises an error
if attention does not reach the end, instead of saving a truncated sample.
Completing the trajectory does not guarantee that every character is legible
or correct: generated labels are conditioning targets, not OCR-verified text.
