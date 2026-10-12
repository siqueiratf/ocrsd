from __future__ import annotations

import argparse
import hashlib
import json
import resource
import time
from collections import Counter
from pathlib import Path

from scipy.optimize import linear_sum_assignment

from src.mapping_assist.analysis import analyze_document
from src.mapping_assist.geometry import iou
from src.mapping_assist.matching import match_candidates
from src.mapping_assist.proposal import build_proposal
from src.semantic.resolve import resolve_field

ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS = {
    "dnv": (ROOT / "documents/dnv/template.pdf", ROOT / "documents/dnv/mappings/mapeamento_seguindo_norma.json"),
    "certidao_nascimento": (ROOT / "documents/certidao_nascimento/certidao_nascimento.pdf",
                            ROOT / "documents/certidao_nascimento/certidao_nascimento_fields.json"),
    "declaracao_obito": (ROOT / "documents/do/do.pdf", None),
}
IOU_THRESHOLD = 0.3


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pairs(proposed: list[dict], reference: list[dict], threshold: float = IOU_THRESHOLD) -> list[tuple[int, int, float]]:
    if not proposed or not reference:
        return []
    scores = [[iou(p["bbox"], r["bbox"]) for r in reference] for p in proposed]
    size = max(len(proposed), len(reference))
    weights = [[0.0] * size for _ in range(size)]
    for i, row in enumerate(scores):
        for j, overlap in enumerate(row):
            if overlap >= threshold:
                weights[i][j] = 1 + overlap
    rows, cols = linear_sum_assignment(weights, maximize=True)
    return [(int(i), int(j), round(scores[i][j], 4)) for i, j in zip(rows, cols)
            if i < len(proposed) and j < len(reference) and scores[i][j] >= threshold]


def score_proposal(proposal: dict, reference: dict, document: str, threshold: float = IOU_THRESHOLD) -> dict:
    proposed = proposal["fields"]
    manual = reference["fields"]
    if proposal["page_size_px"] != reference["page_size_px"]:
        raise ValueError(f"Dimensões diferentes: {proposal['page_size_px']} e {reference['page_size_px']}")
    pairs = _pairs(proposed, manual, threshold)
    matched_proposed = {i for i, _, _ in pairs}
    matched_manual = {j for _, j, _ in pairs}
    semantic = [(i, j, ref.key, proposed[i].get("semantic")) for i, j, _ in pairs
                if (ref := resolve_field(manual[j], document)) is not None]
    correct = sum(key == predicted or isinstance(predicted, dict) and predicted.get("key") == key
                  for _, _, key, predicted in semantic)
    total_semantic = sum(resolve_field(field, document) is not None for field in manual)
    errors = [{"reference": manual[j]["field_id"], "expected": key,
               "predicted": predicted.get("key") if isinstance(predicted, dict) else predicted}
              for _, j, key, predicted in semantic
              if not (key == predicted or isinstance(predicted, dict) and predicted.get("key") == key)]
    return {
        "reference_fields": len(manual), "proposed_fields": len(proposed), "matched_regions": len(pairs),
        "detection_precision": round(len(pairs) / len(proposed), 4) if proposed else None,
        "detection_recall": round(len(pairs) / len(manual), 4) if manual else None,
        "mean_iou_matched": round(sum(overlap for _, _, overlap in pairs) / len(pairs), 4) if pairs else None,
        "iou_threshold": threshold,
        "semantic_reference_fields": total_semantic,
        "semantic_compared": len(semantic),
        "semantic_accuracy_on_matched_regions": round(correct / len(semantic), 4) if semantic else None,
        "semantic_correct_over_reference": round(correct / total_semantic, 4) if total_semantic else None,
        "semantic_errors": errors,
        "unmatched_reference": [field["field_id"] for j, field in enumerate(manual) if j not in matched_manual],
        "unmatched_proposals": [field["field_id"] for i, field in enumerate(proposed) if i not in matched_proposed],
        "region_misses_or_semantic_errors": len(manual) - len(pairs) + len(errors),
        "suggestions_requiring_human_review": len(proposed),
    }


def evaluate(document: str, *, ocr: bool = True) -> dict:
    pdf, mapping = DOCUMENTS[document]
    if not pdf.is_file():
        raise FileNotFoundError(pdf)
    started = time.perf_counter()
    analysis = analyze_document(pdf, ocr=ocr)
    extraction_seconds = time.perf_counter() - started
    started = time.perf_counter()
    matches = match_candidates(analysis, document)
    proposal = build_proposal(analysis, matches, document)
    matching_seconds = time.perf_counter() - started
    statuses = Counter(match.status for match in matches)
    result = {
        "document": document, "pdf_sha256": _digest(pdf), "reference_sha256": _digest(mapping) if mapping else None,
        "page": analysis.page + 1, "source": analysis.source, "page_size_px": proposal["page_size_px"],
        "candidates": len(analysis.candidates), "suggestions": len(proposal["fields"]),
        "statuses": dict(statuses), "warnings": proposal["provenance"]["warnings"],
        "timings_seconds": {"extraction": round(extraction_seconds, 3), "matching_and_conversion": round(matching_seconds, 3),
                            "total": round(extraction_seconds + matching_seconds, 3), "model_inference": 0.0},
        "peak_process_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "provider": "none",
    }
    if mapping is None:
        result["ground_truth"] = None
        result["metrics"] = None
        result["mapping_completion_time"] = None
        return result
    reference = json.loads(mapping.read_text(encoding="utf-8"))
    result["ground_truth"] = "manual"
    result["metrics"] = score_proposal(proposal, reference, document)
    result["mapping_completion_time"] = None
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Avalia o mapeamento assistido contra referências manuais disponíveis.")
    parser.add_argument("--document", choices=(*DOCUMENTS, "all"), default="all")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-ocr", action="store_true")
    args = parser.parse_args()
    documents = DOCUMENTS if args.document == "all" else (args.document,)
    report = {document: evaluate(document, ocr=not args.no_ocr) for document in documents}
    output = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(output + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
