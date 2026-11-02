"""Ground-truth-only report for the DO scenario model / cause-chain library (phase 1, topic 5,
post-verification v2).

Generates N synthetic DO semantic records (no rendering, via the active mapping — see
src/document_types.py) and prints:
- scenario / place / chain distribution
- field fill-rate (of the fields this phase supports)
- duplicate-name rate
- every coherence invariant the independent verifier's review (var/audit/t5v/verification.md)
  required: scenario<->chain<->field48, age/sex gate (>=18), hospital-only chain setting,
  certifier/necropsy/place rules, establishment-only-in-hospital, home-death==residence,
  traffic-never-at-home, CID place-digit, CID/text whitelist, Part II non-duplication,
  date ordering, CNS era, graduate-occupation schooling.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from datetime import date

sys.path.insert(0, ".")
from src.document_types import discover_document_types
from src.semantic.plan import semantic_plan
from src.semantic.record import SemanticRecord
from src.semantic.cause_chains import CAUSE_CHAINS, PART2_POOL

CID_PATTERN = re.compile(r"^[A-Z][0-9]{2}(\.?[0-9X])?$")
CHAINS_BY_ID = {c.chain_id: c for c in CAUSE_CHAINS}
# W00-Y34 codes get a place digit appended dynamically from field 20 (base stored in the
# table); V01-V99 codes are stored COMPLETE (R3) and never touched by the generator.
CID_WHITELIST: dict[str, set[str]] = {}
for _chain in CAUSE_CHAINS:
    for _line in _chain.part1:
        _key = _line.cid[:3] if re.fullmatch(r"[WXY][0-9]{2}", _line.cid) else _line.cid
        CID_WHITELIST.setdefault(_key, set()).update(_line.text_variants)
for _line in PART2_POOL:
    CID_WHITELIST.setdefault(_line.cid, set()).update(_line.text_variants)


def _cid_base(cid: str) -> str:
    return cid[:3] if re.fullmatch(r"[WXY][0-9]{2}\.[0-9]", cid) else cid


def main(n: int = 1000) -> None:
    mapping_path = discover_document_types()["do"].active_mapping
    d = json.load(open(mapping_path, encoding="utf-8"))
    # Mirror the real generation path (src/synthetic_dataset.py): only fields without
    # unsupported_reason are ever "selected" for a real render.
    fields = [f for f in d["fields"] if not f.get("unsupported_reason")]
    plan = semantic_plan(fields, "declaracao_obito")

    manner_counts, scenario_counts, place_counts, chain_counts = Counter(), Counter(), Counter(), Counter()
    names, errors = [], []
    bad = Counter()
    for seed in range(1001, 1001 + n):
        rec = SemanticRecord(seed=seed, context="declaracao_obito")
        try:
            plan.apply(rec)
        except Exception as exc:
            errors.append((seed, str(exc)))
            continue
        v = rec._values
        manner_counts[v["death.cause.manner"]] += 1
        scenario_counts[v["death.scenario"]] += 1
        place_counts[v["death.place.category"]] += 1
        chain_counts[v["death.cause.chain_id"]] += 1
        names.append(v["deceased.name"])
        chain = CHAINS_BY_ID[v["death.cause.chain_id"]]
        hospital = v["death.place.category"] in ("1", "2")

        if v["death.scenario"] not in chain.scenario_ids: bad["B1 scenario/chain mismatch"] += 1
        if chain.manner != v["death.cause.manner"]: bad["B2 manner mismatch"] += 1
        if chain.external_type and chain.external_type != v["death.external.type"]: bad["B1 field48 != chain"] += 1
        if v["deceased.age"] < 18: bad["C minor"] += 1
        if not (chain.min_age <= v["deceased.age"] <= chain.max_age): bad["age outside chain gate"] += 1
        if chain.sex and chain.sex != v["deceased.sex"]: bad["sex mismatch"] += 1
        if chain.setting == "hospital" and not hospital: bad["hospital-only chain outside hospital"] += 1
        manner = v["death.cause.manner"]
        if manner == "Não natural" and v["certifier.role"] != "3": bad["B4 external not IML"] += 1
        if manner == "Não natural" and v["death.autopsy"] != "1": bad["B4 external no necropsy"] += 1
        if manner == "Natural" and v["certifier.role"] == "3": bad["B4 IML on natural"] += 1
        if hospital and v["certifier.role"] == "4": bad["B4 SVO on hospital"] += 1
        if v["certifier.role"] in ("3", "4") and v["death.autopsy"] != "1": bad["B4 IML/SVO no necropsy"] += 1
        if not hospital and v.get("death.place.establishment.name"): bad["B5 establishment outside hospital"] += 1
        if v["death.place.category"] == "1" and not v.get("death.place.establishment.cnes"): bad["hospital no cnes"] += 1
        # R4: field 22 is blank by design for a death in the deceased's own residence (category
        # "3", ~85% of home deaths); only a via-pública death (category "4") must always have one.
        if v["death.place.category"] == "4" and not v.get("deceased.occurrence.address"):
            bad["public no address"] += 1
        if v["death.place.category"] == "3" and v["deceased.residence.municipality"] != v["death.place.municipality"]:
            bad["M1 home death diff municipality"] += 1
        if v["death.scenario"] == "external_traffic" and v["death.place.category"] == "3":
            bad["M1 traffic at home"] += 1
        if v.get("death.external.source") == "Hospital" and not hospital: bad["M1 source hospital wrongly"] += 1
        if "hospital" in (v.get("death.external.description") or "") and not hospital:
            bad["M1 description says hospital wrongly"] += 1
        for letter in "abcd":
            cid, text = v.get(f"death.cause.{letter}_cid"), v.get(f"death.cause.{letter}_text")
            if cid is None:
                continue
            if cid[0] in "WXY":
                expect = {"1": "9", "2": "9", "3": "0", "4": "4"}[v["death.place.category"]]
                if not cid.endswith("." + expect): bad["M2 wrong place digit"] += 1
            if cid[0] == "V" and "." not in cid: bad["R3 V code missing full 4th character"] += 1
            base = _cid_base(cid)
            if base not in CID_WHITELIST or text not in CID_WHITELIST[base]: bad["CID/text not in whitelist"] += 1
        if v["death.date"] < date(2011, 1, 1): bad["M5 date before 2011"] += 1
        if v.get("deceased.cns") and v["death.date"] < date(1997, 1, 1): bad["M5 CNS before 1997"] += 1
        p1_bases = {(v.get(f"death.cause.{letter}_cid") or "").split(".")[0] for letter in "abcd"} - {""}
        for suffix in ("ii1", "ii2"):
            cid = v.get(f"death.cause.{suffix}_cid")
            if cid and cid.split(".")[0] in p1_bases: bad["M2 Part II duplicates Part I"] += 1
        if v.get("death_registration.date") < v["death.date"] or v["deceased.birth_date"] > v["death.date"]:
            bad["date order"] += 1
        occupation = v.get("deceased.occupation")
        if occupation in ("Engenheira", "Advogada", "Professora", "Enfermeiro") and v.get("deceased.schooling") != "Superior completo":
            bad["M4 graduate occupation wrong schooling"] += 1

    total = n - len(errors)
    print(f"mapping: {mapping_path}")
    print(f"generated {total}/{n} (errors={len(errors)})")
    for seed, msg in errors[:5]:
        print("  error seed", seed, msg[:200])
    print("manner:", dict(manner_counts))
    print("scenario:", dict(scenario_counts))
    print("place category:", dict(place_counts))
    print("chains used:", len(chain_counts), "of", len(CAUSE_CHAINS), "available")
    print("unused chains:", [c.chain_id for c in CAUSE_CHAINS if chain_counts[c.chain_id] == 0])
    print("name duplicate rate:", 1 - len(set(names)) / len(names))
    print("--- invariant violations (want all zero) ---")
    if not bad:
        print("NONE")
    for k, c in sorted(bad.items()):
        print(f"{c:6d}  {k}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 1000)
