from __future__ import annotations

import argparse
import importlib.metadata
import json
import shutil
import subprocess
import sys
from pathlib import Path

from src import ocr_bench, pipeline


def preprocess_images(args: argparse.Namespace) -> None:
    import cv2
    from src.preprocess import preprocess_for_ocr

    source = args.input.resolve()
    destination = args.output.resolve()
    if not source.exists():
        raise ValueError(f"Input does not exist: {source}")
    if destination == source or (source.is_dir() and source in destination.parents):
        raise ValueError("Output must be outside the input directory")
    parameters = {}
    mode = args.mode
    if args.preset:
        preset = json.loads(args.preset.read_text(encoding="utf-8"))
        if preset.get("preset_version") != 1:
            raise ValueError("Unsupported preset version")
        mode = preset["mode"]
        parameters = preset["parameters"]
    images = [Path(path) for path in ocr_bench.collect_images(str(source))]
    if not images:
        raise ValueError("No images found")
    for path in images:
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Cannot read image: {path}")
        processed = preprocess_for_ocr(image, mode=mode, **parameters)
        relative = Path(path.name) if source.is_file() else path.relative_to(source)
        target = destination / relative
        if target.exists() and not args.overwrite:
            raise ValueError(f"Output exists: {target}; use --overwrite to replace it")
        target.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(target), processed):
            raise OSError(f"Cannot save image: {target}")
        print(target)


def generate(args: argparse.Namespace) -> None:
    from src.synthetic_dataset import generate_dataset

    mapping = json.loads(args.field_map.read_text(encoding="utf-8"))
    available = {str(field["field_id"]) for field in mapping["fields"]}
    selected = available if args.fields is None else set(args.fields)
    if selected - available:
        raise ValueError(f"Unknown fields: {', '.join(sorted(selected - available))}")
    documents = generate_dataset(
        args.template, args.field_map, selected, None, args.count,
        seed=args.seed, handwriting=not args.printed,
        text_handwriting=not args.printed,
        handwriting_intensity=args.intensity,
        neural_pen=pen_from_args(args, args.handwriting_model)
        if args.handwriting_model != "font" and not args.printed else None,
        realistic_filters=args.realistic_filters,
        filter_preset=args.filter_preset,
    )
    output = documents[0].pdf_path.parent if documents else None
    print(f"Generated {len(documents)} documents in {output}")


def pen_from_args(args: argparse.Namespace, model: str):
    from src.neural_handwriting import NeuralPen

    return NeuralPen(
        model=model, writer=args.writer,
        bias=args.bias if args.bias is not None else (0.75 if model == "sjvasquez" else 2.1),
        color=args.pen_color, width=args.pen_width, pressure=args.pen_pressure,
        dropout=args.pen_dropout, text_policy=args.text_policy,
    )


def handwriting(args: argparse.Namespace) -> None:
    import numpy as np
    from src.neural_handwriting import load_model, prepare_text, render_strokes

    if args.download:
        if args.model == "sjvasquez":
            from src.sj_handwriting import download_weights
            print(download_weights())
        else:
            from src.neural_handwriting import download_weights
            print(download_weights(args.model))
        if args.text is None:
            return
    if not args.text or args.output is None or args.output.suffix.lower() != ".png":
        raise ValueError("Provide --text and --output ending in .png (or use --download)")
    pen = pen_from_args(args, args.model)
    import random
    from dataclasses import replace
    if pen.writer is None:
        pen = replace(pen, writer=random.Random(args.seed).randrange(13 if pen.model == "sjvasquez" else 512))
    text = prepare_text(args.text, pen.text_policy)
    model = load_model(pen.model)
    points = model.sample(text, pen.writer, pen.bias, args.seed)
    image = render_strokes(points, args.width, args.height, pen, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output)
    np.savez(args.output.with_suffix(".strokes.npz"), points=points)
    args.output.with_suffix(".json").write_text(json.dumps({
        **pen.metadata(), "requested_text": args.text, "rendered_text": text,
        "seed": args.seed, "width": args.width, "height": args.height,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(args.output)


def add_pen_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--writer", type=int, help="Style ID: sjvasquez 0-12, aaron1259/wpvnteam 0-511; default random per document")
    parser.add_argument("--bias", type=float, help="Neatness bias, 0-3; defaults: sjvasquez 0.75, aaron1259/wpvnteam 2.1")
    parser.add_argument("--pen-color", default="#193787", help="RGB hex color, e.g. '#193787'")
    parser.add_argument("--pen-width", type=float, default=1.5, help="Stroke width in pixels at the field-map DPI")
    parser.add_argument("--pen-pressure", type=float, default=0.25, help="Pressure variation, 0-1")
    parser.add_argument("--pen-dropout", type=float, default=0.0, help="Intermittent missing ink, 0-1")
    parser.add_argument("--text-policy", choices=("strict", "normalize"), default="strict")


def doctor(_args: argparse.Namespace) -> None:
    import cv2
    import fitz
    import torch
    from src.document_types import discover_document_types

    print(f"Python: {sys.version.split()[0]} ({sys.executable})")
    print(f"OpenCV: {cv2.__version__}; PyMuPDF: {fitz.VersionBind}")
    print(f"PyTorch: {torch.__version__}; CUDA available: {torch.cuda.is_available()}")
    for package in ("easyocr", "transformers", "pytesseract", "paddleocr", "paddlepaddle"):
        try:
            version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            version = "not installed (optional for alternative backends)"
        print(f"{package}: {version}")
    if shutil.which("tesseract"):
        result = subprocess.run(
            ["tesseract", "--list-langs"], capture_output=True, text=True, check=True,
        )
        print(result.stdout.strip())
    else:
        print("Tesseract binary: not installed")
    catalog = discover_document_types()
    print("Document templates: " + (", ".join(catalog) or "none; add documents/<type>/template.pdf"))
    from src.neural_handwriting import WEIGHTS_PATH
    from src.sj_handwriting import ROOT
    for name, path in (("aaron1259", WEIGHTS_PATH), ("wpvnteam", WEIGHTS_PATH), ("sjvasquez", ROOT / "weights.npz")):
        print(f"Handwriting {name}: {'weights present' if path.is_file() else 'not downloaded'}")
    print("Model weights are loaded/downloaded when a backend is first used.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Document processing and synthetic dataset tools.")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser(
        "run", parents=[pipeline.build_parser()], add_help=False,
        help="Align, detect, recognize and redact a PDF",
    )
    run.set_defaults(handler=pipeline.run_from_args)
    benchmark = commands.add_parser(
        "benchmark", parents=[ocr_bench.build_parser()], add_help=False,
        help="Compare OCR backends on images",
    )
    benchmark.set_defaults(handler=ocr_bench.run_from_args)
    preprocess = commands.add_parser("preprocess", help="Preprocess an image or a folder recursively")
    preprocess.add_argument("--input", type=Path, required=True)
    preprocess.add_argument("--output", type=Path, required=True, help="Destination directory")
    preprocess.add_argument("--mode", choices=("traditional", "vlm"), default="traditional")
    preprocess.add_argument("--preset", type=Path, help="JSON preset from the desktop editor; overrides --mode")
    preprocess.add_argument("--overwrite", action="store_true")
    preprocess.set_defaults(handler=preprocess_images)
    dataset = commands.add_parser("generate", help="Generate synthetic PDFs and ground-truth JSON")
    dataset.add_argument("--template", type=Path, required=True)
    dataset.add_argument("--field-map", type=Path, required=True)
    dataset.add_argument("--fields", nargs="+", help="Field IDs; defaults to all mapped fields")
    dataset.add_argument("--count", type=int, default=10)
    dataset.add_argument("--seed", type=int)
    dataset.add_argument("--printed", action="store_true", help="Use printed text instead of handwriting")
    dataset.add_argument("--intensity", type=float, default=0.65)
    dataset.add_argument("--handwriting-model", choices=("font", "sjvasquez", "aaron1259", "wpvnteam"), default="font")
    dataset.add_argument("--realistic-filters", action="store_true", help="Apply scan-like light, shadow, noise and compression filters")
    dataset.add_argument("--filter-preset", choices=("light", "medium", "strong"), default="medium")
    add_pen_arguments(dataset)
    dataset.set_defaults(handler=generate)
    preview = commands.add_parser("handwriting", help="Download a local handwriting model or render a cursive PNG")
    preview.add_argument("--model", choices=("sjvasquez", "aaron1259", "wpvnteam"), default="sjvasquez")
    preview.add_argument("--download", action="store_true", help="Download and verify model weights")
    preview.add_argument("--text")
    preview.add_argument("--output", type=Path)
    preview.add_argument("--seed", type=int, default=42)
    preview.add_argument("--width", type=int, default=900)
    preview.add_argument("--height", type=int, default=160)
    add_pen_arguments(preview)
    preview.set_defaults(handler=handwriting)
    models = commands.add_parser("models", help="List OCR backend names")
    models.set_defaults(handler=lambda _args: print("\n".join(
        ocr_bench.BACKEND_FACTORIES
    )))
    diagnostic = commands.add_parser("doctor", help="Check installed dependencies and local templates")
    diagnostic.set_defaults(handler=doctor)
    return parser


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.handler(args)
    except (OSError, ValueError, TypeError, KeyError, ImportError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
