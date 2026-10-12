import argparse
import json
from pathlib import Path

import cv2
import fitz
import numpy as np

from src.config import PipelineConfig
from src.document_types import DOCUMENTS_ROOT
from src.pipeline import run_pipeline
from src.semantic.registry import REGISTRY
from src.synthetic_dataset import generate_dataset


CASES = (
    ("dnv", "template.pdf", "mappings/mapeamento_seguindo_norma.json"),
    ("certidao_nascimento", "certidao_nascimento.pdf", "certidao_nascimento_fields.json"),
)


def expected_content(truth: dict) -> set[str]:
    filled = set()
    snapshot = truth.get("semantic_record") or {}
    for field in truth["fields"]:
        identifier = str(field["field_id"])
        if field.get("value") != "":
            if field.get("blank") or field.get("value") is None:
                raise AssertionError(f"Ground truth inconsistente: {identifier}")
            filled.add(identifier)
            continue
        semantic = field.get("semantic") or {}
        key = semantic.get("key")
        definition = REGISTRY.get(key)
        if not (field.get("blank") is True and definition and definition.optional
                and "value" in semantic and semantic["value"] is None
                and key in snapshot and snapshot[key] is None):
            raise AssertionError(f"Campo vazio sem justificativa semântica: {identifier}")
    return filled


def verify_document(key: str, template_name: str, mapping_name: str,
                    output: Path, seed: int) -> dict:
    root = DOCUMENTS_ROOT / key
    template = root / template_name
    mapping_path = root / mapping_name
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    selected = {str(field["field_id"]) for field in mapping["fields"]}
    generated = generate_dataset(
        template, mapping_path, selected, output / key / "dataset", 1, seed=seed,
    )[0]
    truth = json.loads(generated.json_path.read_text(encoding="utf-8"))
    if {str(field["field_id"]) for field in truth["fields"]} != selected:
        raise AssertionError(f"{key}: campos ausentes no documento sintético")
    expected = expected_content(truth)
    with fitz.open(generated.pdf_path) as document, fitz.open(template) as blank:
        if len(document) != 1:
            raise AssertionError(f"{key}: número de páginas incorreto")
        filled_pixmap = document[0].get_pixmap(dpi=mapping["dpi"], colorspace=fitz.csRGB)
        blank_pixmap = blank[0].get_pixmap(dpi=mapping["dpi"], colorspace=fitz.csRGB)
        shape = (filled_pixmap.height, filled_pixmap.width, filled_pixmap.n)
        filled_pixels = np.frombuffer(filled_pixmap.samples, dtype=np.uint8).reshape(shape)
        blank_pixels = np.frombuffer(blank_pixmap.samples, dtype=np.uint8).reshape(shape)
        for field in truth["fields"]:
            x1, y1, x2, y2 = field["bbox"]
            changed = not np.array_equal(filled_pixels[y1:y2, x1:x2], blank_pixels[y1:y2, x1:x2])
            if str(field["field_id"]) in expected and not changed:
                raise AssertionError(f"{key}: campo não renderizado: {field['field_id']}")
            if str(field["field_id"]) not in expected and changed:
                raise AssertionError(f"{key}: campo semanticamente vazio foi renderizado: {field['field_id']}")
    processed = output / key / "pipeline"
    run_pipeline(
        str(generated.pdf_path), str(template), str(processed),
        PipelineConfig(dpi=mapping["dpi"]), document_id=key,
        field_map_path=str(mapping_path), skip_ocr=True,
    )
    manifest = json.loads((processed / f"{key}.json").read_text(encoding="utf-8"))
    if not manifest["alignment"]["success"]:
        raise AssertionError(f"{key}: alinhamento falhou")
    entries = manifest["fields"]
    if {str(field["field_id"]) for field in entries} != selected:
        raise AssertionError(f"{key}: campos ausentes no processamento")
    filled = [field for field in entries if field["filled"]]
    expected_redacted = expected | {str(field["field_id"]) for field in mapping["fields"] if field.get("always_redact")}
    detected = {str(field["field_id"]) for field in filled}
    if detected != expected_redacted:
        raise AssertionError(f"{key}: detecção divergente: ausentes={sorted(expected_redacted - detected)}, "
                             f"inesperados={sorted(detected - expected_redacted)}")
    for field in entries:
        if cv2.imread(str(processed / field["crop_path"])) is None:
            raise AssertionError(f"{key}: recorte ausente: {field['field_id']}")
    with fitz.open(processed / f"{key}.pdf") as redacted:
        if len(redacted) != 1 or redacted[0].get_text().strip():
            raise AssertionError(f"{key}: PDF ocultado inválido")
        pixmap = redacted[0].get_pixmap(dpi=mapping["dpi"], colorspace=fitz.csRGB)
        pixels = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
            pixmap.height, pixmap.width, pixmap.n,
        )
        for field in filled:
            x1, y1, x2, y2 = field["bbox"]
            center = pixels[(y1 + y2) // 2, (x1 + x2) // 2, :3]
            if not np.array_equal(center, [255, 0, 0]):
                raise AssertionError(f"{key}: campo sem ocultação: {field['field_id']}")
    return {"type": key, "seed": seed, "fields": len(entries),
            "expected_filled": len(expected), "semantic_blanks": len(selected - expected),
            "detected": len(filled), "pdf": str(generated.pdf_path)}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Gera documentos sintéticos e verifica o processamento completo.")
    parser.add_argument("--output", type=Path, default=Path("output/smoke"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    results = [verify_document(*case, args.output, args.seed) for case in CASES]
    report = args.output / "validation.json"
    report.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Validação concluída: {report}")


if __name__ == "__main__":
    main()
