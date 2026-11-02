import json
import re
import unittest
from collections import Counter

from src.semantic.cause_chains import CAUSE_CHAINS, PART2_POOL, UNIT_RANK, chains_for_scenario, format_interval
from src.semantic.death_scenarios import ADULT_SCENARIOS, applicable_field_groups, pick_scenario
from src.semantic.plan import semantic_plan
from src.semantic.record import SemanticRecord
import random

CID_PATTERN = re.compile(r"^[A-Z][0-9]{2}(\.?[0-9X])?$")
SCENARIO_IDS = {s.scenario_id for s in ADULT_SCENARIOS}
_UNIT_MINUTES = {"minutos": 1, "horas": 60, "dias": 1440, "semanas": 10080, "meses": 43200, "anos": 525600}
CHAINS_BY_ID = {c.chain_id: c for c in CAUSE_CHAINS}


def _interval_minutes(text: str) -> int:
    n, unit = text.split(" ", 1)
    unit = {"minuto": "minutos", "hora": "horas", "dia": "dias", "semana": "semanas", "mês": "meses",
           "ano": "anos"}.get(unit, unit)
    return int(n) * _UNIT_MINUTES[unit]


def _do_plan():
    mapping = json.load(open("documents/do/mappings/mapeamento_do_v2.json", encoding="utf-8"))
    fields = [f for f in mapping["fields"] if not f.get("unsupported_reason")]
    return semantic_plan(fields, "declaracao_obito")


class CauseChainLibraryTests(unittest.TestCase):
    def test_every_cid_is_well_formed(self):
        for chain in CAUSE_CHAINS:
            for line in chain.part1:
                self.assertTrue(CID_PATTERN.fullmatch(line.cid) or re.fullmatch(r"[VWXY][0-9]{2}", line.cid),
                                f"{chain.chain_id}: {line.cid}")
        for line in PART2_POOL:
            self.assertTrue(CID_PATTERN.fullmatch(line.cid), line.cid)

    def test_part1_intervals_are_non_decreasing_from_a_to_d(self):
        # a (immediate) must be the shortest interval, d (underlying) the longest; lines
        # sharing a `group` id are allowed to tie since they share one sampled draw.
        for chain in CAUSE_CHAINS:
            ranks = [UNIT_RANK[line.unit] for line in chain.part1]
            for i in range(len(ranks) - 1):
                self.assertLessEqual(ranks[i], ranks[i + 1], f"{chain.chain_id} line {i}")

    def test_part1_has_one_to_four_lines(self):
        for chain in CAUSE_CHAINS:
            self.assertTrue(1 <= len(chain.part1) <= 4, chain.chain_id)

    def test_every_chain_declares_a_known_scenario(self):
        for chain in CAUSE_CHAINS:
            self.assertTrue(chain.scenario_ids, chain.chain_id)
            self.assertTrue(set(chain.scenario_ids) <= SCENARIO_IDS, chain.chain_id)

    def test_external_chains_declare_a_single_scenario_and_matching_manner_type(self):
        scenario_manner = {s.scenario_id: s.manner for s in ADULT_SCENARIOS}
        for chain in CAUSE_CHAINS:
            if chain.category == "external":
                self.assertEqual(len(chain.scenario_ids), 1, chain.chain_id)
                self.assertEqual(chain.manner, "Não natural", chain.chain_id)
                self.assertIn(chain.external_type, {"Acidente", "Homicídio", "Suicídio"}, chain.chain_id)
                self.assertEqual(scenario_manner[chain.scenario_ids[0]], "Não natural", chain.chain_id)
            else:
                self.assertEqual(chain.manner, "Natural", chain.chain_id)
                self.assertIsNone(chain.external_type, chain.chain_id)
                self.assertTrue(set(chain.scenario_ids) <= {"natural_hospital", "natural_home"}, chain.chain_id)

    def test_chains_for_scenario_matches_declared_scenario_ids(self):
        for scenario_id in SCENARIO_IDS:
            matches = chains_for_scenario(scenario_id)
            for chain in matches:
                self.assertIn(scenario_id, chain.scenario_ids)
            # every scenario must have at least one eligible chain, and at least one that
            # does not require a hospital (so a non-hospital place is never impossible)
            self.assertTrue(matches, scenario_id)
            self.assertTrue(any(c.setting != "hospital" for c in matches), scenario_id)

    def test_min_age_is_at_least_18_phase_1(self):
        for chain in CAUSE_CHAINS:
            self.assertGreaterEqual(chain.min_age, 18, chain.chain_id)

    def test_no_duplicate_part1_cid_within_a_chain(self):
        for chain in CAUSE_CHAINS:
            cids = [line.cid for line in chain.part1]
            self.assertEqual(len(cids), len(set(cids)), chain.chain_id)

    def test_format_interval_singular_plural(self):
        self.assertEqual(format_interval("horas", 1), "1 hora")
        self.assertEqual(format_interval("horas", 2), "2 horas")
        self.assertEqual(format_interval("anos", 1), "1 ano")
        self.assertEqual(format_interval("anos", 10), "10 anos")


class ScenarioModelTests(unittest.TestCase):
    def test_scenario_weights_sum_positive_and_cover_natural_and_external(self):
        manners = {s.manner for s in ADULT_SCENARIOS}
        self.assertEqual(manners, {"Natural", "Não natural"})
        self.assertTrue(all(s.weight > 0 for s in ADULT_SCENARIOS))

    def test_place_weights_sum_to_100(self):
        for scenario in ADULT_SCENARIOS:
            self.assertAlmostEqual(sum(scenario.place_weights.values()), 100, places=5, msg=scenario.scenario_id)
            self.assertTrue(set(scenario.place_weights) <= {"1", "2", "3", "4"}, scenario.scenario_id)

    def test_traffic_scenario_never_weights_home(self):
        traffic = next(s for s in ADULT_SCENARIOS if s.scenario_id == "external_traffic")
        self.assertNotIn("3", traffic.place_weights)

    def test_pick_scenario_is_deterministic_given_rng_state(self):
        a = pick_scenario(random.Random(1234))
        b = pick_scenario(random.Random(1234))
        self.assertEqual(a.scenario_id, b.scenario_id)

    def test_applicable_field_groups_include_external_only_when_not_natural(self):
        natural = next(s for s in ADULT_SCENARIOS if s.manner == "Natural")
        external = next(s for s in ADULT_SCENARIOS if s.manner == "Não natural")
        self.assertNotIn("external_cause", applicable_field_groups(natural, hospital=True))
        self.assertIn("external_cause", applicable_field_groups(external, hospital=True))

    def test_applicable_field_groups_pick_occurrence_group_by_hospital_flag(self):
        scenario = ADULT_SCENARIOS[0]
        self.assertIn("hospital_occurrence", applicable_field_groups(scenario, hospital=True))
        self.assertIn("home_or_public_occurrence", applicable_field_groups(scenario, hospital=False))


class PostVerificationPropertyTests(unittest.TestCase):
    """Property tests over >=1000 seeds for the independent verifier's re-verification fixes
    (R1-R5, var/audit/t5v/verification.md)."""
    N = 1200
    START = 90000

    @classmethod
    def setUpClass(cls):
        plan = _do_plan()
        cls.records = []
        for seed in range(cls.START, cls.START + cls.N):
            record = SemanticRecord(seed=seed, context="declaracao_obito")
            plan.apply(record)
            cls.records.append(record._values)

    def test_r1_drawn_interval_values_are_non_decreasing_in_minutes(self):
        # Compares the DRAWN values (not just the declared units) converted to a common unit —
        # the old test only checked unit order, which let overlapping ranges invert (R1).
        violations = []
        for values in self.records:
            minutes = [_interval_minutes(values[f"death.cause.{letter}_interval"]) for letter in "abcd"
                      if values.get(f"death.cause.{letter}_interval")]
            if minutes != sorted(minutes):
                violations.append((values["death.cause.chain_id"], minutes))
        self.assertEqual(violations, [])

    def test_r2_chronic_home_chains_mostly_have_medical_assistance(self):
        chronic_home = [v for v in self.records if v["death.cause.manner"] == "Natural"
                        and v["death.place.category"] not in ("1", "2")
                        and CHAINS_BY_ID[v["death.cause.chain_id"]].course == "chronic"]
        self.assertGreater(len(chronic_home), 20)
        with_assistance = sum(1 for v in chronic_home if v["death.medical_assistance"] == "1")
        self.assertGreaterEqual(with_assistance / len(chronic_home), 0.8)
        # ... and so are certified by an attending/substitute doctor, not the SVO.
        svo = sum(1 for v in chronic_home if v["certifier.role"] == "4")
        self.assertLessEqual(svo / len(chronic_home), 0.2)

    def test_r3_v_codes_are_stored_complete_and_never_place_coded(self):
        for values in self.records:
            for letter in "abcd":
                cid = values.get(f"death.cause.{letter}_cid")
                if cid and cid[0] == "V":
                    self.assertIn(".", cid, cid)
                    chain = CHAINS_BY_ID[values["death.cause.chain_id"]]
                    line = chain.part1["abcd".index(letter)]
                    self.assertEqual(cid, line.cid, "V codes are never place-coded by the generator")

    def test_r4_own_residence_blanks_field_22_and_other_address_uses_profile_street(self):
        own_count = other_count = 0
        for values in self.records:
            if values["death.place.category"] != "3":
                continue
            address = values.get("deceased.occurrence.address")
            description = values.get("death.external.description") or ""
            if address is None:
                own_count += 1
                self.assertNotIn("em domicílio", description)
            else:
                other_count += 1
                self.assertNotIn("próprio domicílio", description)
                # Uses the same address generator as the residence (a municipality profile
                # street with its own matching CEP/neighborhood), never the old 5-item generic
                # list (whose CEP/neighborhood never matched the street's own profile).
                # item 5: a municipality offers several streets, so `address` may be any of a
                # profile's street names, and its CEP must match THAT street (not necessarily
                # the profile's primary one).
                from src.semantic.data import LOCATION_PROFILES
                street_info = {name: (cep, neighborhood) for p in LOCATION_PROFILES for name, cep, neighborhood in p.get("streets", ())}
                profile = next(p for p in LOCATION_PROFILES
                              if address == p["street"] or address in street_info)
                street_cep, street_neighborhood = street_info.get(address, (profile["cep"], profile["neighborhood"]))
                self.assertEqual(values["deceased.occurrence.cep"], street_cep)
                self.assertEqual(values["deceased.occurrence.neighborhood"], street_neighborhood)
        self.assertGreater(own_count, 0)
        self.assertGreater(other_count, 0)

    def test_r5_ai_proposed_age_always_yields_a_coherent_chain(self):
        # Simulates assisted.py's flow: an exact age is required (as the AI would propose, now
        # clamped to [18, 102] in assisted.SLOTS) before the record is resolved; death() must
        # pick a scenario/chain pair whose gate contains it, every time (R5).
        plan = _do_plan()
        for age in (18, 25, 40, 60, 80, 102):
            for seed in range(1, 30):
                record = SemanticRecord(seed=seed, context="declaracao_obito")
                record.require("deceased.age", (age,), "idade proposta pela IA")
                plan.apply(record)
                self.assertEqual(record._values["deceased.age"], age)
                chain = CHAINS_BY_ID[record._values["death.cause.chain_id"]]
                self.assertLessEqual(chain.min_age, age)
                self.assertLessEqual(age, chain.max_age)


if __name__ == "__main__":
    unittest.main()
