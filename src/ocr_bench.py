import argparse
import glob
import json
import os
import sys
import time
from dataclasses import asdict, dataclass

import cv2
import numpy as np


@dataclass
class OcrPrediction:
    text: str
    confidence: float | None
    latency_ms: float


class OcrBackend:
    name = "base"

    def recognize(self, image: np.ndarray) -> OcrPrediction:
        raise NotImplementedError


class EasyOcrBackend(OcrBackend):
    name = "easyocr"

    def __init__(self, languages: tuple[str, ...] = ("en", "pt")):
        import easyocr
        import torch

        gpu = torch.cuda.is_available()
        self.reader = easyocr.Reader(list(languages), gpu=gpu, verbose=False)

    def recognize(self, image: np.ndarray) -> OcrPrediction:
        t0 = time.perf_counter()
        detections = self.reader.readtext(image, detail=1, paragraph=False)
        latency_ms = (time.perf_counter() - t0) * 1000
        if not detections:
            return OcrPrediction("", 0.0, latency_ms)
        texts = [text for _, text, _ in detections]
        confidences = [float(conf) for _, _, conf in detections]
        return OcrPrediction(" ".join(texts).strip(), sum(confidences) / len(confidences), latency_ms)


class TrOcrBackend(OcrBackend):
    def __init__(self, model_id: str):
        self.name = "trocr-" + model_id.split("/")[-1].replace("trocr-", "")
        import torch
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        self._torch = torch
        self.processor = TrOCRProcessor.from_pretrained(model_id)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_id)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()

    def recognize(self, image: np.ndarray) -> OcrPrediction:
        from PIL import Image

        t0 = time.perf_counter()
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        pixel_values = self.processor(images=Image.fromarray(rgb), return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device)

        with self._torch.no_grad():
            out = self.model.generate(
                pixel_values, max_new_tokens=64, output_scores=True, return_dict_in_generate=True
            )
        text = self.processor.batch_decode(out.sequences, skip_special_tokens=True)[0]

        confidence = None
        if out.scores:
            token_probs = [self._torch.softmax(step[0], dim=-1).max().item() for step in out.scores]
            confidence = sum(token_probs) / len(token_probs)

        latency_ms = (time.perf_counter() - t0) * 1000
        return OcrPrediction(text.strip(), confidence, latency_ms)


class TesseractBackend(OcrBackend):
    name = "tesseract"

    def __init__(self, lang: str = "por+eng", psm: int = 7):
        import pytesseract

        self._pytesseract = pytesseract
        self.lang = lang
        self.config = f"--psm {psm}"

    def recognize(self, image: np.ndarray) -> OcrPrediction:
        t0 = time.perf_counter()
        data = self._pytesseract.image_to_data(
            image, lang=self.lang, config=self.config, output_type=self._pytesseract.Output.DICT
        )
        latency_ms = (time.perf_counter() - t0) * 1000

        words = [(text, conf) for text, conf in zip(data["text"], data["conf"]) if text.strip()]
        if not words:
            return OcrPrediction("", 0.0, latency_ms)
        text = " ".join(word for word, _ in words)
        confidences = [float(conf) for _, conf in words if float(conf) >= 0]
        confidence = (sum(confidences) / len(confidences) / 100.0) if confidences else 0.0
        return OcrPrediction(text.strip(), confidence, latency_ms)


class PaddleOcrBackend(OcrBackend):
    name = "paddleocr"

    def __init__(self, lang: str = "pt"):
        from paddleocr import PaddleOCR

        self.reader = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            lang=lang,
        )

    def recognize(self, image: np.ndarray) -> OcrPrediction:
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.ndim == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        t0 = time.perf_counter()
        results = self.reader.predict(image)
        latency_ms = (time.perf_counter() - t0) * 1000

        texts: list[str] = []
        scores: list[float] = []
        for res in results:
            texts.extend(res["rec_texts"])
            scores.extend(res["rec_scores"])

        if not texts:
            return OcrPrediction("", 0.0, latency_ms)
        confidence = sum(scores) / len(scores) if scores else None
        return OcrPrediction(" ".join(texts).strip(), confidence, latency_ms)


BACKEND_FACTORIES = {
    "easyocr": lambda: EasyOcrBackend(),
    "trocr-handwritten": lambda: TrOcrBackend("microsoft/trocr-base-handwritten"),
    "trocr-printed": lambda: TrOcrBackend("microsoft/trocr-base-printed"),
    "trocr-large-handwritten": lambda: TrOcrBackend("microsoft/trocr-large-handwritten"),
    "tesseract": lambda: TesseractBackend(),
    "paddleocr": lambda: PaddleOcrBackend(),
}

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")


def collect_images(input_path: str) -> list[str]:
    if os.path.isfile(input_path):
        return [input_path]
    paths = [
        p for p in glob.glob(os.path.join(input_path, "**", "*"), recursive=True)
        if p.lower().endswith(IMAGE_EXTENSIONS)
    ]
    return sorted(paths)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run OCR on a set of images through one or more model backends and "
                    "save a side-by-side comparison to JSON."
    )
    parser.add_argument("--input", required=True, help="Directory of images, or a single image file")
    parser.add_argument(
        "--models", required=True,
        help="Comma-separated backend names. Available: " + ", ".join(BACKEND_FACTORIES)
    )
    parser.add_argument("--output", required=True, help="Path to write the comparison JSON")
    parser.add_argument(
        "--preprocess", choices=["none", "traditional", "vlm"], default="none",
        help="Apply src/preprocess.py before OCR: 'traditional' (ink isolation + denoise + "
             "CLAHE + deskew + binarize + pad, for TrOCR/CRNN/Tesseract/EasyOCR-style models) "
             "or 'vlm' (same but stays color, for Gemini/GPT-4o-style models). Default 'none'."
    )
    parser.add_argument(
        "--save-preprocessed", default=None,
        help="Optional directory to also save the preprocessed images to, for visual inspection."
    )
    return parser


def parse_args(argv=None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def run_from_args(args: argparse.Namespace) -> None:
    images = collect_images(args.input)
    if not images:
        print(f"No images found in {args.input}", file=sys.stderr)
        sys.exit(1)

    model_names = [name.strip() for name in args.models.split(",") if name.strip()]
    unknown = [name for name in model_names if name not in BACKEND_FACTORIES]
    if unknown or not model_names:
        print(f"Unknown model(s): {', '.join(unknown)}. Available: {', '.join(BACKEND_FACTORIES)}",
              file=sys.stderr)
        sys.exit(1)

    backends: dict[str, OcrBackend | None] = {}
    for name in model_names:
        print(f"Loading {name}...")
        try:
            backends[name] = BACKEND_FACTORIES[name]()
        except Exception as exc:
            print(f"  Unavailable: {exc}", file=sys.stderr)
            backends[name] = None

    if args.save_preprocessed:
        os.makedirs(args.save_preprocessed, exist_ok=True)

    results = []
    for image_path in images:
        image = cv2.imread(image_path)
        if image is None:
            print(f"  Skipping unreadable image: {image_path}", file=sys.stderr)
            continue

        if args.preprocess != "none":
            from src import preprocess
            image = preprocess.preprocess_for_ocr(image, mode=args.preprocess)
            if args.save_preprocessed:
                out_name = (os.path.basename(image_path) if os.path.isfile(args.input)
                            else os.path.relpath(image_path, args.input).replace(os.sep, "_"))
                cv2.imwrite(os.path.join(args.save_preprocessed, out_name), image)


        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        image_name = (os.path.basename(image_path) if os.path.isfile(args.input)
                      else os.path.relpath(image_path, args.input))
        entry = {"image": image_name, "predictions": {}}
        summary_parts = []
        for name, backend in backends.items():
            if backend is None:
                entry["predictions"][name] = {"error": "backend unavailable"}
                summary_parts.append(f"{name}=<unavailable>")
                continue
            try:
                prediction = backend.recognize(image)
                entry["predictions"][name] = asdict(prediction)
                summary_parts.append(f"{name}={prediction.text!r}")
            except Exception as exc:
                entry["predictions"][name] = {"error": str(exc)}
                summary_parts.append(f"{name}=<error>")
        results.append(entry)
        print(f"{entry['image']}: " + " | ".join(summary_parts))

    output = {
        "input": args.input,
        "models": model_names,
        "preprocess": args.preprocess,
        "image_count": len(results),
        "results": results,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nSaved {len(results)} images x {len(model_names)} models to {args.output}")
    print("\nSummary (successful recognitions / avg latency):")
    for name in model_names:
        latencies = [
            e["predictions"][name]["latency_ms"] for e in results
            if "latency_ms" in e["predictions"][name]
        ]
        ok = len(latencies)
        avg = sum(latencies) / ok if ok else 0.0
        print(f"  {name}: {ok}/{len(results)} ok, avg {avg:.0f}ms")

    if not any("latency_ms" in prediction
               for entry in results for prediction in entry["predictions"].values()):
        raise SystemExit(1)


def main(argv=None) -> None:
    run_from_args(parse_args(argv))


if __name__ == "__main__":
    main()
