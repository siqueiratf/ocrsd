import json
import random
import re
import tempfile
import unittest
from datetime import date
from pathlib import Path

import fitz

from src.document_types import discover_document_types
from src.semantic.cause_chains import CAUSE_CHAINS, CHILD_PART2_POOL, PART2_POOL
from src.semantic.constraints import completed_years
from src.semantic.plan import semantic_plan
from src.semantic.record import SemanticRecord
from src.synthetic_dataset import generate_document


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "documents/do/do.pdf"
MAPPING = discover_document_types()["do"].active_mapping

CID_PATTERN = re.compile(r"^[A-Z][0-9]{2}(\.?[0-9X])?$")
CHAINS_BY_ID = {chain.chain_id: chain for chain in CAUSE_CHAINS}
# Whitelist of real CID-10 codes this phase actually uses, mapped to the canonical Portuguese
# text(s) they may appear with. W/X/Y codes are keyed by their 3-character base (the DO's own
# field 20 supplies the 4th "place" digit at render time — see cause_chains._place_coded_cid);
# V01-V99 codes are stored COMPLETE in the table (R3) and keyed as-is.
CID_WHITELIST: dict[str, set[str]] = {}
for _chain in CAUSE_CHAINS:
    for _line in _chain.part1:
        _key = _line.cid[:3] if re.fullmatch(r"[WXY][0-9]{2}", _line.cid) else _line.cid
        CID_WHITELIST.setdefault(_key, set()).update(_line.text_variants)
for _line in PART2_POOL:
    CID_WHITELIST.setdefault(_line.cid, set()).update(_line.text_variants)
for _line in CHILD_PART2_POOL:
    CID_WHITELIST.setdefault(_line.cid, set()).update(_line.text_variants)


def _cid_base(cid: str) -> str:
    return cid[:3] if re.fullmatch(r"[WXY][0-9]{2}\.[0-9]", cid) else cid


class DOModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mapping = json.loads(MAPPING.read_text(encoding="utf-8"))

    def test_official_reference_geometry_and_generation_scope(self):
        with fitz.open(TEMPLATE) as pdf:
            self.assertEqual(len(pdf), 1)
            self.assertEqual(tuple(pdf[0].rect[2:]), (612, 792))
        self.assertIn("Apêndice A", self.mapping["reference_manual"])
        self.assertIn("Reprodução", self.mapping["template_provenance"])
        plan = semantic_plan(self.mapping["fields"], "declaracao_obito")
        self.assertFalse(plan.issues)
        ids = [field["field_id"] for field in self.mapping["fields"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len(ids), 103)
        unsupported = {field["field_id"] for field in self.mapping["fields"] if field.get("unsupported_reason")}
        # Phase 1 (topic 5, post-verification v2) added deterministic support for the
        # cause-of-death chain, the external-cause block and the certificate signature.
        # Phase 2 (v3) added the fetal/infant age-unit block, the maternal/obstetric block
        # (27-37) and the Part II intervals. Block IX ("Localidade sem médico", 58/59) remains
        # out of scope — it is only filled on the real form when no physician certifies the
        # death, a separate rare scenario this phase does not implement (see verifier B3).
        self.assertTrue({"58", "59_a", "59_b", "9_ignorado"} <= unsupported)
        self.assertFalse({"40_a_causa", "40_a_cid", "47", "48", "9_meses", "9_dias", "9_horas", "9_minutos",
                          "27", "31", "35", "36"} & unsupported)
        self.assertEqual(len(next(field for field in self.mapping["fields"]
                                  if field["field_id"] == "9_ignorado")["elements"]), 1)
        for field in self.mapping["fields"]:
            x1, y1, x2, y2 = field["bbox"]
            self.assertTrue(0 <= x1 < x2 <= 2550 and 0 <= y1 < y2 <= 3300, field["field_id"])
            for element in field.get("elements", []):
                ex1, ey1, ex2, ey2 = element["bbox"]
                self.assertTrue(0 <= ex1 < ex2 <= 2550 and 0 <= ey1 < ey2 <= 3300, field["field_id"])
            if field["field_id"] not in plan.bindings:
                self.assertTrue(field.get("unsupported_reason"), field["field_id"])

    def test_supported_do_fields_generate_consistent_records(self):
        selected = {field["field_id"] for field in self.mapping["fields"] if not field.get("unsupported_reason")}
        with tempfile.TemporaryDirectory() as temp:
            for seed in range(10):
                output = generate_document(TEMPLATE, self.mapping, selected, Path(temp), str(seed),
                                           random.Random(seed), handwriting=False, text_handwriting=False)
                with fitz.open(output.pdf_path) as pdf:
                    self.assertEqual(len(pdf), 1)
                truth = json.loads(output.json_path.read_text(encoding="utf-8"))
                values = {field["field_id"]: field["value"] for field in truth["fields"]}
                record = truth["semantic_record"]
                born = date.fromisoformat(record["deceased.birth_date"])
                died = date.fromisoformat(record["death.date"])
                registered = date.fromisoformat(record["death_registration.date"])
                self.assertLessEqual(born, died)
                self.assertLessEqual(died, registered)
                age_group = record.get("death.age_group", "adult")
                if age_group == "adult":
                    self.assertEqual(record["deceased.age"], completed_years(born, died))
                else:
                    self.assertIsNone(record["deceased.age"])
                self.assertEqual(values["2"], died.strftime("%d%m%Y"))
                self.assertEqual(values["8"], born.strftime("%d%m%Y"))
                self.assertEqual(values["46"], died.strftime("%d%m%Y"))
                self.assertEqual(values["55"], registered.strftime("%d%m%Y"))
                self.assertEqual(values["18"], record["deceased.residence.municipality"])
                self.assertEqual(values["19"], record["deceased.residence.uf"])
                self.assertEqual(values["25"], record["death.place.municipality"])
                self.assertEqual(values["26"], record["death.place.uf"])
                self.assertEqual(values["1"], "1" if age_group == "fetal" else "2")
                self.assertIn(values["20"], {"1", "2", "3", "4"})
                hospital = values["20"] in {"1", "2"}
                if hospital:
                    self.assertEqual(values["38"], "1")
                self.assertIn(values["39"], {"1", "2"})
                self.assertIn(values["43"], {"1", "2", "3", "4"})
                if age_group == "adult":
                    self.assertGreaterEqual(record["deceased.age"], 18)
                # Part I line "a" (immediate cause) is always filled with a well-formed CID.
                self.assertTrue(values["40_a_causa"])
                self.assertRegex(values["40_a_cid"], r"^[A-Z][0-9]{2}(\.?[0-9X])?$")
                external = record["death.cause.manner"] == "Não natural"
                self.assertEqual(bool(values.get("48")), external)
                # block IX never renders (B3): there is no doctor-less scenario in phase 1.
                self.assertEqual(values.get("58", ""), "")
                self.assertEqual(values.get("59_a", ""), "")
                self.assertEqual(values.get("59_b", ""), "")


class DOPropertyTests(unittest.TestCase):
    """Property tests over many seeds for the independent verifier's B1-B5/M1/M2/C invariants.

    These go through ``semantic_plan(...).apply`` directly (ground truth only, no rendering —
    the same fields that are actually selected for a real DO, i.e. excluding unsupported_reason
    fields) so 1000+ seeds run in well under a second and can be exercised on every test run.
    """
    N = 1200
    SEED_START = 30000

    @classmethod
    def setUpClass(cls):
        mapping = json.loads(MAPPING.read_text(encoding="utf-8"))
        fields = [field for field in mapping["fields"] if not field.get("unsupported_reason")]
        plan = semantic_plan(fields, "declaracao_obito")
        cls.records = []
        for seed in range(cls.SEED_START, cls.SEED_START + cls.N):
            record = SemanticRecord(seed=seed, context="declaracao_obito")
            plan.apply(record)
            cls.records.append(dict(record._values))

    def test_chain_always_declared_for_the_drawn_scenario(self):
        for v in self.records:
            chain = CHAINS_BY_ID[v["death.cause.chain_id"]]
            with self.subTest(seed=v):
                self.assertIn(v["death.scenario"], chain.scenario_ids)
                self.assertEqual(chain.manner, v["death.cause.manner"])
                if chain.external_type is not None:
                    self.assertEqual(chain.external_type, v["death.external.type"])

    def test_age_sex_always_inside_the_chosen_chain_gate_and_at_least_18(self):
        for v in self.records:
            chain = CHAINS_BY_ID[v["death.cause.chain_id"]]
            if chain.age_unit == "anos":
                # Phase 2: only "anos"-gated chains (adult, 18+; child, 1-17) have an integer
                # deceased.age at all — fetal/infant export None (see generators.deceased).
                self.assertLessEqual(chain.min_age, v["deceased.age"])
                self.assertLessEqual(v["deceased.age"], chain.max_age)
                if v["death.age_group"] == "adult":
                    self.assertGreaterEqual(v["deceased.age"], 18)
            else:
                self.assertIsNone(v["deceased.age"])
            if chain.sex:
                self.assertEqual(chain.sex, v["deceased.sex"])

    def test_hospital_only_chains_are_never_drawn_outside_a_hospital(self):
        for v in self.records:
            chain = CHAINS_BY_ID[v["death.cause.chain_id"]]
            if chain.setting == "hospital":
                self.assertIn(v["death.place.category"], ("1", "2"))

    def test_certifier_necropsy_and_place_follow_the_scenario(self):
        for v in self.records:
            hospital = v["death.place.category"] in ("1", "2")
            manner = v["death.cause.manner"]
            if manner == "Não natural":
                self.assertEqual(v["certifier.role"], "3", v)
                self.assertEqual(v["death.autopsy"], "1", v)
            else:
                self.assertNotEqual(v["certifier.role"], "3", v)
            if v["certifier.role"] == "4":
                self.assertFalse(hospital, v)
                self.assertEqual(v["death.autopsy"], "1", v)
            if hospital:
                self.assertNotEqual(v["certifier.role"], "4", v)
            if v["certifier.role"] in ("3", "4"):
                self.assertEqual(v["death.autopsy"], "1", v)

    def test_establishment_only_set_for_hospital_deaths(self):
        for v in self.records:
            hospital = v["death.place.category"] in ("1", "2")
            if hospital:
                self.assertIsNotNone(v["death.place.establishment.name"])
                if v["death.place.category"] == "1":
                    self.assertIsNotNone(v["death.place.establishment.cnes"])
            else:
                self.assertIsNone(v["death.place.establishment.name"])
                self.assertIsNone(v["death.place.establishment.cnes"])

    def test_occurrence_address_matches_residence_when_death_is_at_home(self):
        # R4: a death in the deceased's OWN residence leaves field 22 blank outright (the form
        # says it's for an occurrence outside the establishment OR the residence); only the
        # minority at a different address fills it, using the same profile-based generator as
        # the residence (so CEP/neighborhood always match that street's own profile).
        own, other = 0, 0
        for v in self.records:
            if v["death.place.category"] == "3":
                self.assertEqual(v["deceased.residence.municipality"], v["death.place.municipality"], v)
                if v["deceased.occurrence.address"] is None:
                    own += 1
                else:
                    other += 1
        self.assertGreater(own, 0)
        self.assertGreater(own, other)

    def test_traffic_deaths_are_never_at_home(self):
        for v in self.records:
            if v["death.scenario"] == "external_traffic":
                self.assertNotEqual(v["death.place.category"], "3", v)

    def test_cid_place_digit_matches_field_20(self):
        # Only W00-Y34 use the 4th character for the place of occurrence (R3); V01-V99 uses it
        # for the victim's role/accident mechanism and is stored complete, never touched here.
        digit_by_category = {"1": "9", "2": "9", "3": "0", "4": "4"}
        for v in self.records:
            for letter in "abcd":
                cid = v.get(f"death.cause.{letter}_cid")
                if cid and cid[0] in "WXY":
                    self.assertTrue(cid.endswith("." + digit_by_category[v["death.place.category"]]), (v, cid))
                if cid and cid[0] == "V":
                    self.assertIn(".", cid, (v, cid))

    def test_cid_and_text_come_from_the_whitelist(self):
        for v in self.records:
            for letter in "abcd":
                cid, text = v.get(f"death.cause.{letter}_cid"), v.get(f"death.cause.{letter}_text")
                if cid is None:
                    self.assertIsNone(text)
                    continue
                self.assertRegex(cid, CID_PATTERN)
                base = _cid_base(cid)
                self.assertIn(base, CID_WHITELIST, (v["death.cause.chain_id"], cid))
                self.assertIn(text, CID_WHITELIST[base], (v["death.cause.chain_id"], cid, text))
            for suffix in ("ii1", "ii2"):
                cid, text = v.get(f"death.cause.{suffix}_cid"), v.get(f"death.cause.{suffix}_text")
                if cid is None:
                    self.assertIsNone(text)
                    continue
                self.assertIn(cid, CID_WHITELIST, (suffix, cid))
                self.assertIn(text, CID_WHITELIST[cid], (suffix, cid, text))

    def test_interval_order_and_part2_never_duplicates_part1(self):
        for v in self.records:
            p1_bases = set()
            for letter in "abcd":
                cid = v.get(f"death.cause.{letter}_cid")
                if cid:
                    p1_bases.add(cid.split(".")[0])
            for suffix in ("ii1", "ii2"):
                cid = v.get(f"death.cause.{suffix}_cid")
                if cid:
                    self.assertNotIn(cid.split(".")[0], p1_bases, (v["death.cause.chain_id"], cid))

    def test_date_order_and_cns_era(self):
        for v in self.records:
            self.assertGreaterEqual(v["death.date"], date(2011, 1, 1))
            self.assertLessEqual(v["deceased.birth_date"], v["death.date"])
            self.assertLessEqual(v["death.date"], v["death_registration.date"])
            if v.get("deceased.cns"):
                self.assertGreaterEqual(v["death.date"], date(1997, 1, 1))

    def test_graduate_occupations_have_matching_schooling(self):
        for v in self.records:
            if v["deceased.occupation"] in ("Engenheira", "Advogada", "Professora", "Enfermeiro"):
                self.assertEqual(v["deceased.schooling"], "Superior completo", v)


if __name__ == "__main__":
    unittest.main()
