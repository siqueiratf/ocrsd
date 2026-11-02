#!/usr/bin/env python3
"""Quality gate (topic 6, item 10): run the writing audit's deterministic --quick subset
and compare a small set of key aggregates against a stored reference, failing on a
regression beyond tolerance.

Usage (from repo root):
    PYTHONPATH=. python scripts/quality_gate.py            # run + compare, exit 1 on regression
    PYTHONPATH=. python scripts/quality_gate.py --update   # run + overwrite the reference

Also runnable as `make quality` (see Makefile). The reference lives at
tests/fixtures/quality_reference.json -- a small JSON of aggregates only, never images,
so it's cheap to diff in review. To intentionally accept a regression (e.g. after a
deliberate behaviour change), rerun with --update and review+commit the new reference
like any other test fixture.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REFERENCE_PATH = ROOT / "tests" / "fixtures" / "quality_reference.json"
LABEL = "quality_gate"

# Absolute tolerance per metric (these are fractions/ratios/counts; see reference.json's
# own "_source" field for provenance of the stored numbers).
TOLERANCES = {
    "mean_cer_loose": 0.08,
    "mean_cer_loose_digits_only": 0.10,
    "excessive_spacing_count": 2,
    "mean_left_margin_h": 0.12,
    "mean_digit_height_ratio": 0.08,
    "right_overflow_count": 1,
}
FILL_RATE_TOLERANCE_PCT = 2.0
# Coherence issue counts are expected to be 0 (see check_coherence/report.md); any
# non-zero count on a document that was previously 0 is a correctness regression (data
# became internally inconsistent), so there is no numeric tolerance -- a single new
# issue fails the gate.
COHERENCE_COUNT_TOLERANCE = 0


def extract_aggregates(summary: dict) -> dict:
    agg = {"per_preset": {}, "per_document_fill_pct": {}, "coherence_issue_counts": {}}
    for uid, info in summary.get("per_preset", {}).items():
        agg["per_preset"][uid] = {k: info.get(k) for k in TOLERANCES}
    for dkey, info in summary.get("per_document", {}).items():
        agg["per_document_fill_pct"][dkey] = info.get("fill_rate_over_selectable_pct")
    for dkey, issues in (summary.get("coherence") or {}).items():
        if dkey.startswith("_"):
            continue
        agg["coherence_issue_counts"][dkey] = sum((issues or {}).values())
    return agg


def run_quick() -> dict:
    out_dir = ROOT / "var" / "audit" / LABEL
    cmd = [sys.executable, str(ROOT / "scripts" / "writing_audit.py"), "--label", LABEL, "--quick"]
    subprocess.run(cmd, cwd=ROOT, check=True)
    return json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))


def compare(current: dict, reference: dict) -> list[str]:
    problems = []
    for uid, ref_metrics in reference.get("per_preset", {}).items():
        cur_metrics = current.get("per_preset", {}).get(uid)
        if cur_metrics is None:
            problems.append(f"preset {uid} missing from current run")
            continue
        for metric, ref_val in ref_metrics.items():
            cur_val = cur_metrics.get(metric)
            if ref_val is None or cur_val is None:
                continue
            tol = TOLERANCES[metric]
            # "worse" direction: CER/spacing/overflow UP is bad; left_margin/digit_height
            # moving further from the reference in either direction is flagged, since both
            # were deliberately tuned into a target range by earlier topics.
            if metric in ("mean_cer_loose", "mean_cer_loose_digits_only", "excessive_spacing_count", "right_overflow_count"):
                if cur_val - ref_val > tol:
                    problems.append(f"{uid}.{metric}: {ref_val} -> {cur_val} (regression > {tol})")
            else:
                if abs(cur_val - ref_val) > tol:
                    problems.append(f"{uid}.{metric}: {ref_val} -> {cur_val} (drift > {tol})")
    for dkey, ref_pct in reference.get("per_document_fill_pct", {}).items():
        cur_pct = current.get("per_document_fill_pct", {}).get(dkey)
        if ref_pct is not None and cur_pct is not None and ref_pct - cur_pct > FILL_RATE_TOLERANCE_PCT:
            problems.append(f"{dkey} fill_rate: {ref_pct}% -> {cur_pct}% (regression > {FILL_RATE_TOLERANCE_PCT}pp)")
    # Coherence: a document with 0 issues at reference time that now has any must fail;
    # a document that already had issues at reference time must not get worse.
    for dkey, cur_count in current.get("coherence_issue_counts", {}).items():
        ref_count = reference.get("coherence_issue_counts", {}).get(dkey, 0)
        if cur_count - ref_count > COHERENCE_COUNT_TOLERANCE:
            problems.append(f"{dkey} coherence_issue_count: {ref_count} -> {cur_count} (regression)")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--update", action="store_true", help="Overwrite the stored reference with the current run.")
    args = parser.parse_args()

    summary = run_quick()
    aggregates = extract_aggregates(summary)

    if args.update:
        aggregates["_source"] = ("Generated by `python scripts/quality_gate.py --update` from the `--quick` "
                                  "audit subset (see scripts/writing_audit.py:QUICK_DOCUMENTS/QUICK_PRESETS). "
                                  "Review the diff before committing.")
        REFERENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
        REFERENCE_PATH.write_text(json.dumps(aggregates, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Updated {REFERENCE_PATH}")
        return

    if not REFERENCE_PATH.is_file():
        print(f"No reference at {REFERENCE_PATH}. Run with --update to create one.", file=sys.stderr)
        sys.exit(1)
    reference = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
    problems = compare(aggregates, reference)
    if problems:
        print("QUALITY GATE FAILED:")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)
    print("Quality gate passed: no regression beyond tolerance vs tests/fixtures/quality_reference.json")


if __name__ == "__main__":
    main()
