from __future__ import annotations

import argparse
import json
import resource
import time
from collections import Counter
from pathlib import Path

from src.mapping_assist.analysis import analyze_document
from src.mapping_assist.geometry import iou
from src.mapping_assist.matching import match_candidates
from src.mapping_assist.proposal import build_proposal
from src.mapping_assist.providers import NullAssistant, assistant_from_settings
from src.semantic.plan import semantic_plan
from src.semantic.resolve import document_context, resolve_field

ROOT = Path(__file__).resolve().parents[1]
PRESETS = {
    "dnv": (ROOT / "documents/dnv/template.pdf", ROOT / "documents/dnv/mappings/mapeamento_seguindo_norma.json"),
    "certidao_nascimento": (ROOT / "documents/certidao_nascimento/certidao_nascimento.pdf",
                            ROOT / "documents/certidao_nascimento/certidao_nascimento_fields.json"),
    "declaracao_obito": (ROOT / "documents/do/do.pdf", None),
}
THRESHOLDS = (0.3, 0.5)
SIZE_TOLERANCE = 2


def pairs(proposed: list[dict], reference: list[dict], threshold: float) -> list[tuple[int, int, float]]:
    scored = sorted(((iou(p["bbox"], r["bbox"]), i, j) for i, p in enumerate(proposed) for j, r in enumerate(reference)),
                    key=lambda item: (-item[0], item[1], item[2]))
    used_p, used_r, found = set(), set(), []
    for overlap, i, j in scored:
        if overlap < threshold:
            break
        if i not in used_p and j not in used_r:
            used_p.add(i)
            used_r.add(j)
            found.append((i, j, round(overlap, 4)))
    return found


def proposed_key(field: dict) -> str | None:
    semantic = field.get("semantic")
    return semantic.get("key") if isinstance(semantic, dict) else semantic or None


def reference_keys(reference: dict, document: str | None) -> dict[str, str | None]:
    fields = reference["fields"]
    try:
        decisions = semantic_plan(fields, document or reference.get("template_id")).decisions
    except Exception:
        decisions = {}
    keys = {}
    for field in fields:
        decision = decisions.get(str(field["field_id"]))
        binding = decision.binding if decision else resolve_field(field, document)
        keys[str(field["field_id"])] = binding.key if binding else None
    return keys


def _ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def score(proposal: dict, reference: dict, document: str | None, threshold: float) -> dict:
    sizes = list(proposal["page_size_px"]), list(reference["page_size_px"])
    if len(sizes[0]) != 2 or len(sizes[1]) != 2 or any(abs(a - b) > SIZE_TOLERANCE for a, b in zip(*sizes)):
        raise ValueError(f"Dimensões diferentes: {proposal['page_size_px']} e {reference['page_size_px']}")
    proposed, manual = proposal["fields"], reference["fields"]
    matched = pairs(proposed, manual, threshold)
    keys = reference_keys(reference, document)
    composite = [field for field in manual if field.get("elements")]
    element_total = sum(len(field["elements"]) for field in composite)
    element_hits = sum(len(pairs(proposed[i].get("elements") or [], manual[j]["elements"], threshold))
                       for i, j, _ in matched if manual[j].get("elements"))
    semantic, errors, corrections = [], [], 0
    for i, j, overlap in matched:
        expected, predicted = keys.get(str(manual[j]["field_id"])), proposed_key(proposed[i])
        wrong = expected is not None and predicted != expected
        if expected is not None:
            semantic.append(not wrong)
        if wrong:
            errors.append({"reference": manual[j]["field_id"], "expected": expected, "predicted": predicted})
        corrections += wrong or overlap < 0.5
    unmatched_reference = [manual[j]["field_id"] for j in sorted(set(range(len(manual))) - {j for _, j, _ in matched})]
    false_positives = [proposed[i]["field_id"] for i in sorted(set(range(len(proposed))) - {i for i, _, _ in matched})]
    precision, recall = _ratio(len(matched), len(proposed)), _ratio(len(matched), len(manual))
    return {
        "iou_threshold": threshold, "reference_fields": len(manual), "proposed_fields": len(proposed), "matched_fields": len(matched),
        "precision": precision, "recall": recall,
        "f1": round(2 * precision * recall / (precision + recall), 4) if precision and recall else None,
        "mean_iou_matched": round(sum(overlap for _, _, overlap in matched) / len(matched), 4) if matched else None,
        "element_recall": _ratio(element_hits, element_total), "reference_elements": element_total,
        "semantic_reference_fields": sum(key is not None for key in keys.values()), "semantic_compared": len(semantic),
        "semantic_accuracy": _ratio(sum(semantic), len(semantic)),
        "matched_status_counts": dict(Counter(proposed[i].get("suggestion", {}).get("match", "none") for i, _, _ in matched)),
        "manual_corrections": len(unmatched_reference) + len(false_positives) + corrections,
        "semantic_errors": errors, "unmatched_reference": unmatched_reference, "false_positives": false_positives,
    }


def evaluate(template: Path, mapping: Path | None = None, document: str | None = None, *, ocr: bool = True,
             assistant=None) -> dict:
    context = document_context(document)
    started = time.perf_counter()
    analysis = analyze_document(template, ocr=ocr)
    analyzed = time.perf_counter()
    budget = None
    if assistant is None:
        try:
            from src.web.settings import Settings

            settings = Settings()
            assistant, budget = assistant_from_settings(settings, settings.storage / "ai-cache"), settings.ai_budget
        except ImportError:
            assistant = NullAssistant()
    status = assistant.status()
    matches = match_candidates(analysis, context, assistant if status.available else None, status=status, budget=budget)
    matched = time.perf_counter()
    proposal = build_proposal(analysis, matches, context)
    finished = time.perf_counter()
    fields = proposal["fields"]
    result = {
        "template": str(template), "document": context, "source": analysis.source, "page_size_px": proposal["page_size_px"],
        "provider": status.provider if status.available else "none", "model": status.model if status.available else None,
        "candidates": len(analysis.candidates), "proposed_fields": len(fields),
        "status_counts": dict(Counter(field["suggestion"]["match"] for field in fields)),
        "data_types": dict(Counter(field["data_type"] for field in fields)),
        "sections": dict(Counter(field.get("bloco") or "-" for field in fields)),
        "top_labels": [{"label": field["label"], "key": proposed_key(field)} for field in fields if proposed_key(field)][:15],
        "warnings": len(proposal["provenance"]["warnings"]),
        "timings_seconds": {"analysis_stages": dict(analysis.timings), "analysis": round(analyzed - started, 3),
                            "matching": round(matched - analyzed, 3), "proposal": round(finished - matched, 3),
                            "total": round(finished - started, 3)},
        "peak_rss_mib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
    }
    if mapping is not None:
        reference = json.loads(Path(mapping).read_text(encoding="utf-8"))
        result["mapping"] = str(mapping)
        result["metrics"] = {f"iou_{threshold}": score(proposal, reference, context, threshold) for threshold in THRESHOLDS}
    return result


def summary(result: dict) -> str:
    lines = [f"{result['template']} ({result['document'] or 'sem contexto'}, {result['source']}, provedor {result['provider']})",
             f"candidatos {result['candidates']}, propostas {result['proposed_fields']}, status {result['status_counts']}",
             f"tempo total {result['timings_seconds']['total']} s, pico RSS {result['peak_rss_mib']} MiB"]
    for name, metrics in (result.get("metrics") or {}).items():
        lines.append(f"{name}: precisão {metrics['precision']}, revocação {metrics['recall']}, IoU médio {metrics['mean_iou_matched']}, "
                     f"revocação de elementos {metrics['element_recall']}, acurácia semântica {metrics['semantic_accuracy']} "
                     f"({metrics['semantic_compared']}), correções manuais {metrics['manual_corrections']}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Avalia o mapeamento assistido de um template contra um mapeamento de referência.")
    parser.add_argument("--template", type=Path)
    parser.add_argument("--mapping", type=Path)
    parser.add_argument("--document")
    parser.add_argument("--no-ocr", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    template, mapping = args.template, args.mapping
    if template is None:
        preset = PRESETS.get(document_context(args.document) or "")
        if preset is None:
            parser.error("informe --template ou um --document com referência conhecida")
        template, mapping = preset[0], mapping or preset[1]
    result = evaluate(template, mapping, args.document, ocr=not args.no_ocr)
    print(json.dumps(result, ensure_ascii=False, indent=2) if args.json else summary(result))


if __name__ == "__main__":
    main()
