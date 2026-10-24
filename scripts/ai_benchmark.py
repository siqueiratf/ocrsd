from __future__ import annotations

import argparse
import json
import statistics
import tempfile
import time
from collections import Counter
from pathlib import Path

from src.document_types import discover_document_types
from src.field_rules import apply_shared_rules
from src.mapping_assist.providers import AssistantError, OpenAICompatibleAssistant
from src.semantic.assisted import AssistedGenerationError, AssistedRecords
from src.synthetic_dataset import generate_dataset

ROOT = Path(__file__).resolve().parents[1]


def trim_rules(text: str, limit: int) -> str:
    text = text.strip()
    if limit <= 0 or len(text) <= limit:
        return text
    cut = text.rfind("\n\n", 0, limit)
    if cut < limit * 0.4:
        cut = limit
    return text[:cut].strip()


def document_keys(documents_root: Path, requested: str | None) -> list[str]:
    types = discover_document_types(documents_root)
    if not types:
        raise SystemExit(f"Nenhum documento encontrado em {documents_root}")
    if requested in (None, "all"):
        return sorted(types)
    if requested not in types:
        raise SystemExit(f"Documento desconhecido: {requested}. Disponíveis: {', '.join(sorted(types))}")
    return [requested]


def benchmark_document(key: str, document, assistant, *, count: int, seed_start: int, attempts: int,
                       temperature: float, max_fields_per_request: int, rules_chars: int) -> dict:
    definition = json.loads(document.active_mapping.read_text(encoding="utf-8"))
    resolved = apply_shared_rules(definition)
    fields = [field for field in resolved["fields"] if not field.get("unsupported_reason")]
    selected = {str(field["field_id"]) for field in fields}
    rules_path = document.root / "regras.md"
    rules_text = trim_rules(rules_path.read_text(encoding="utf-8"), rules_chars) if rules_path.is_file() else None
    document_meta = {"nome": document.label, "orgao": (document.orgao or {}).get("nome"), "descricao": document.description}
    requests, latencies, rejected_reasons = 0, [], Counter()
    valid_values, fallback_values, failures = 0, 0, 0
    started = time.monotonic()
    for index in range(count):
        factory = AssistedRecords(assistant, temperature=temperature, attempts=attempts, fields=fields,
                                  dpi=resolved.get("dpi", 300), document_meta=document_meta, rules_text=rules_text,
                                  max_fields_per_request=max_fields_per_request)
        attempts_log = []
        with tempfile.TemporaryDirectory(prefix="ocrsd-ai-benchmark-") as directory:
            output = Path(directory) / "out"
            try:
                generated = generate_dataset(document.template, document.active_mapping, selected, output, 1, seed_start + index,
                                             handwriting=False, text_handwriting=False, record_factory=factory)
            except AssistedGenerationError as exc:
                failures += 1
                attempts_log = exc.attempts
            else:
                truth = json.loads(generated[0].json_path.read_text(encoding="utf-8"))
                attempts_log = (truth.get("semantic_provenance") or {}).get("attempts") or []
        for entry in attempts_log:
            requests += 1
            if entry.get("seconds") is not None:
                latencies.append(entry["seconds"])
            for item in entry.get("rejected") or []:
                rejected_reasons[item["reason"]] += 1
        if factory.documents:
            summary = factory.documents[0]
            valid_values += len(summary.get("ai_keys") or []) + len(summary.get("ai_fields") or [])
            fallback_values += len(summary.get("fallback_fields") or [])
    elapsed = time.monotonic() - started
    return {"document": key, "requests": requests, "latency_avg_seconds": round(statistics.mean(latencies), 3) if latencies else None,
           "latency_max_seconds": round(max(latencies), 3) if latencies else None, "documents_requested": count,
           "documents_failed": failures, "valid_ai_values": valid_values, "fallback_values": fallback_values,
           "rejected_reasons": dict(rejected_reasons.most_common(10)), "elapsed_seconds": round(elapsed, 2)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Mede latência e taxa de aceitação da geração assistida por IA")
    parser.add_argument("--document", default="all", help="Chave do documento (padrão: all, todos os documentos padrão)")
    parser.add_argument("--documents-root", default=str(ROOT / "documents"))
    parser.add_argument("--model", default="qwen3:4b")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434/v1")
    parser.add_argument("--count", type=int, default=5, help="Documentos a gerar por chave")
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--max-fields-per-request", type=int, default=24)
    parser.add_argument("--rules-chars", type=int, default=1800)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", default=None, help="Caminho para gravar o relatório em JSON")
    args = parser.parse_args()

    try:
        assistant = OpenAICompatibleAssistant(args.base_url, args.model, timeout=args.timeout)
    except ValueError as exc:
        raise SystemExit(f"Configuração inválida: {exc}")
    status = assistant.status()
    if not status.available:
        raise SystemExit(f"Nenhum servidor de IA respondeu em {args.base_url} ({status.detail}). "
                         "Instale e inicie um servidor compatível (ex.: Ollama) antes de medir.")

    documents_root = Path(args.documents_root)
    types = discover_document_types(documents_root)
    reports = []
    for key in document_keys(documents_root, args.document):
        print(f"Medindo {key}...", flush=True)
        try:
            report = benchmark_document(key, types[key], assistant, count=args.count, seed_start=args.seed,
                                        attempts=args.attempts, temperature=args.temperature,
                                        max_fields_per_request=args.max_fields_per_request, rules_chars=args.rules_chars)
        except AssistantError as exc:
            report = {"document": key, "error": str(exc)}
        reports.append(report)
        print(f"  requisições={report.get('requests')} latência_media={report.get('latency_avg_seconds')}s "
             f"válidos={report.get('valid_ai_values')} fallback={report.get('fallback_values')} "
             f"falhas={report.get('documents_failed')}")

    result = {"model": args.model, "base_url": args.base_url, "documents": reports}
    if args.out:
        Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Relatório gravado em {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
