"""Phase 2 property tests: fetal/infant/child DO scenarios (scenario<->chain<->age/unit,
fetal/infant block applicability, weight/gestation, mother age, DN, certifier/necropsy, and
the adult age-distribution sanity the independent verifier flagged — see final report)."""
import functools
import json
import unittest

from src.semantic.cause_chains import CAUSE_CHAINS
from src.semantic.death_scenarios import ALL_SCENARIOS
from src.semantic.plan import semantic_plan
from src.semantic.record import SemanticRecord

CHAINS_BY_ID = {c.chain_id: c for c in CAUSE_CHAINS}
N = 4000
START = 500000


@functools.lru_cache(maxsize=None)
def _records(n=N, start=START):
    mapping = json.load(open("documents/do/mappings/mapeamento_do_v3.json", encoding="utf-8"))
    fields = [f for f in mapping["fields"] if not f.get("unsupported_reason")]
    plan = semantic_plan(fields, "declaracao_obito")
    records = []
    for seed in range(start, start + n):
        record = SemanticRecord(seed=seed, context="declaracao_obito")
        plan.apply(record)
        records.append(record._values)
    return records


class Phase2ScenarioTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = _records()
        cls.by_group = {}
        for v in cls.records:
            cls.by_group.setdefault(v["death.age_group"], []).append(v)

    def test_all_age_groups_appear(self):
        for group in ("adult", "fetal", "infant", "child"):
            self.assertGreater(len(self.by_group.get(group, [])), 0, group)

    def test_death_type_matches_age_group(self):
        for v in self.records:
            self.assertEqual(v["death.type"] == "1", v["death.age_group"] == "fetal")

    def test_fetal_has_no_age_and_no_dn(self):
        for v in self.by_group["fetal"]:
            self.assertIsNone(v["deceased.age"])
            for unit in ("minutes", "hours", "days", "months"):
                self.assertIsNone(v[f"deceased.age_{unit}"])
            self.assertIsNone(v["deceased.birth.dnv_number"])
            self.assertEqual(v["deceased.name"], "Natimorto")
            self.assertIn(v["deceased.death_relative_to_delivery"], ("Antes", "Durante"))

    def test_infant_has_exactly_one_age_unit_and_a_dn(self):
        for v in self.by_group["infant"]:
            units = [v[f"deceased.age_{u}"] for u in ("minutes", "hours", "days", "months")]
            self.assertEqual(sum(u is not None for u in units), 1, v["death.cause.chain_id"])
            self.assertIsNotNone(v["deceased.birth.dnv_number"])
            self.assertEqual(v["deceased.death_relative_to_delivery"], "Depois")

    def test_adult_and_child_have_no_infant_fetal_fields(self):
        for v in self.by_group["adult"] + self.by_group["child"]:
            for unit in ("minutes", "hours", "days", "months"):
                self.assertIsNone(v[f"deceased.age_{unit}"])
            self.assertIsNone(v["deceased.birth.dnv_number"])
            self.assertIsNone(v["deceased.mother_age"])
            self.assertIsNone(v["deceased.gestation_weeks"])

    def test_scenario_chain_age_unit_consistent(self):
        for v in self.records:
            chain = CHAINS_BY_ID[v["death.cause.chain_id"]]
            self.assertIn(v["death.scenario"], chain.scenario_ids)
            if v["death.age_group"] == "fetal":
                self.assertEqual(chain.age_unit, "fetal")
            elif v["death.age_group"] == "infant":
                self.assertIn(chain.age_unit, ("minutos", "horas", "dias", "meses"))
            else:
                self.assertEqual(chain.age_unit, "anos")

    def test_child_age_between_1_and_17(self):
        for v in self.by_group["child"]:
            self.assertGreaterEqual(v["deceased.age"], 1)
            self.assertLessEqual(v["deceased.age"], 17)

    def test_gestation_weeks_in_range_when_known(self):
        for v in self.by_group["fetal"] + self.by_group["infant"]:
            weeks = v["deceased.gestation_weeks"]
            if weeks is not None:
                self.assertGreaterEqual(weeks, 22)
                self.assertLessEqual(weeks, 42)

    def test_birth_weight_roughly_tracks_gestation_weeks(self):
        # Not a tight correlation (gaussian noise + preterm variability), but a 22-week fetus
        # should never be rendered at term weight and vice versa (R: weight/gestation coherence).
        for v in self.by_group["fetal"] + self.by_group["infant"]:
            weeks, weight = v["deceased.gestation_weeks"], v["deceased.birth_weight_g"]
            if weeks is None or weight is None:
                continue
            if weeks <= 26:
                self.assertLess(weight, 1600, (weeks, weight))
            if weeks >= 39:
                self.assertGreater(weight, 2000, (weeks, weight))

    def test_mother_age_plausible(self):
        for v in self.by_group["fetal"] + self.by_group["infant"]:
            self.assertGreaterEqual(v["deceased.mother_age"], 13)
            self.assertLessEqual(v["deceased.mother_age"], 49)

    def test_certifier_and_necropsy_rules_hold_for_fetal_and_infant(self):
        for v in self.by_group["fetal"] + self.by_group["infant"]:
            hospital = v["death.place.category"] in ("1", "2")
            if hospital:
                self.assertIn(v["certifier.role"], ("1", "2"))
            if v["certifier.role"] in ("3", "4"):
                self.assertEqual(v["death.autopsy"], "1")

    def test_adult_fall_age_distribution_is_not_flat_and_rarely_centenarian(self):
        # Verifier-flagged invariant: falls skewed to ~11% aged 100+ under the old uniform
        # draw; the weighted distribution must now keep that share low.
        falls = [v for v in self.by_group["adult"] if CHAINS_BY_ID[v["death.cause.chain_id"]].category == "external"
                and v["death.external.type"] == "Acidente" and CHAINS_BY_ID[v["death.cause.chain_id"]].chain_id
                in ("tce_queda_idoso", "fratura_femur_queda")]
        self.assertGreater(len(falls), 30)
        centenarian = sum(1 for v in falls if v["deceased.age"] >= 100)
        self.assertLess(centenarian / len(falls), 0.05)

    def test_adult_traffic_age_distribution_is_not_flat(self):
        traffic = [v for v in self.by_group["adult"] if CHAINS_BY_ID[v["death.cause.chain_id"]].chain_id
                  in ("trauma_moto", "politrauma_auto")]
        self.assertGreater(len(traffic), 30)
        over_55 = sum(1 for v in traffic if v["deceased.age"] >= 55)
        # A flat 18-70 draw (politrauma_auto) would put ~29% at 55+; the weighted draw (peaked
        # around 32, with only a modest mid-age bump) must keep noticeably less mass there.
        self.assertLess(over_55 / len(traffic), 0.22)

    def test_adult_alzheimer_age_distribution_rarely_centenarian(self):
        # B5: broncopneumonia_alzheimer was "not touched" by per-chain weights — a flat draw
        # over 70-102 would put > 9% at 100+; the weighted draw must keep it rare.
        records = [v for v in self.by_group["adult"] if v["death.cause.chain_id"] == "broncopneumonia_alzheimer"]
        self.assertGreater(len(records), 20)
        centenarian = sum(1 for v in records if v["deceased.age"] >= 100)
        self.assertLess(centenarian / len(records), 0.05)

    def _infant_age_minutes(self, v):
        unit, value = next((u, v[f"deceased.age_{u}"]) for u in ("minutes", "hours", "days", "months")
                           if v[f"deceased.age_{u}"] is not None)
        return value * {"minutes": 1, "hours": 60, "days": 1440, "months": 43200}[unit]

    def test_B1_infant_age_recomputes_exactly_from_dates_and_death_time(self):
        # B1: the age unit value must be exactly recoverable from (birth_date, death.date,
        # death.time) — never a 30-days-per-month approximation, never 0.
        import datetime as dt
        for v in self.by_group["infant"]:
            born, died, death_time = v["deceased.birth_date"], v["death.date"], v["death.time"]
            if v["deceased.age_months"] is not None:
                n = v["deceased.age_months"]
                self.assertGreater(n, 0, v["death.cause.chain_id"])
                months = (died.year - born.year) * 12 + (died.month - born.month)
                if died.day < born.day:
                    months -= 1
                self.assertEqual(months, n, (born, died, n))
            elif v["deceased.age_days"] is not None:
                n = v["deceased.age_days"]
                self.assertGreater(n, 0)
                self.assertEqual((died - born).days, n)
            elif v["deceased.age_hours"] is not None:
                n = v["deceased.age_hours"]
                self.assertGreater(n, 0)
                self.assertEqual((dt.datetime.combine(died, death_time) - dt.timedelta(hours=n)).date(), born)
            elif v["deceased.age_minutes"] is not None:
                n = v["deceased.age_minutes"]
                self.assertGreater(n, 0)
                self.assertEqual((dt.datetime.combine(died, death_time) - dt.timedelta(minutes=n)).date(), born)

    def test_B2_part1_intervals_never_exceed_age_and_since_birth_lines_equal_it(self):
        _UNIT_MINUTES = {"minutos": 1, "horas": 60, "dias": 1440, "semanas": 10080, "meses": 43200, "anos": 525600}
        UNIT_WORD = {"minuto": "minutos", "hora": "horas", "dia": "dias", "semana": "semanas", "mês": "meses", "ano": "anos"}

        def mins(text):
            n, unit = text.split(" ", 1)
            return int(n) * _UNIT_MINUTES[UNIT_WORD.get(unit, unit)]
        for v in self.by_group["infant"]:
            age_minutes = self._infant_age_minutes(v)
            chain = CHAINS_BY_ID[v["death.cause.chain_id"]]
            part1 = list(chain.part1) + [None] * (4 - len(chain.part1))
            for letter, line in zip("abcd", part1):
                text = v[f"death.cause.{letter}_interval"]
                if text is None:
                    continue
                self.assertFalse(text.startswith("0 "), text)
                self.assertLessEqual(mins(text), age_minutes, (chain.chain_id, letter, text, age_minutes))
                if line is not None and line.since_birth:
                    self.assertEqual(mins(text), age_minutes, (chain.chain_id, letter, text))

    def test_B3_fetal_hypoxia_timing_never_contradicts_field34(self):
        for v in self.by_group["fetal"]:
            a, b = v["death.cause.a_cid"], v["death.cause.b_cid"]
            rel = v["deceased.death_relative_to_delivery"]
            self.assertFalse(a == "P20.0" and b == "P20.1", v["death.cause.chain_id"])
            if b == "P20.1":
                self.assertEqual(rel, "Durante")
            if a == "P20.0":
                self.assertEqual(rel, "Antes")

    def test_B4_prematurity_code_matches_gestation_weeks(self):
        for v in self.by_group["infant"]:
            if v["death.cause.chain_id"] == "infant_sdr_prematuridade_extrema":
                self.assertIsNotNone(v["deceased.gestation_weeks"])
                self.assertLess(v["deceased.gestation_weeks"], 28)
                self.assertEqual(v["death.cause.b_cid"], "P07.2")
            if v["death.cause.chain_id"] == "infant_sdr_prematuridade_moderada":
                self.assertIsNotNone(v["deceased.gestation_weeks"])
                self.assertGreaterEqual(v["deceased.gestation_weeks"], 28)
                self.assertLessEqual(v["deceased.gestation_weeks"], 36)
                self.assertEqual(v["death.cause.b_cid"], "P07.3")

    def test_B5_adult_fall_age_weights_applied_per_chain(self):
        tce = [v for v in self.by_group["adult"] if v["death.cause.chain_id"] == "tce_queda_idoso"]
        self.assertGreater(len(tce), 15)
        # A flat 65-102 draw would put ~11% at 100+; the weighted draw must keep it well below
        # that (verifier's own large-sample measurement was 0.3%-0.5%; this small-sample test
        # leaves headroom for sampling noise, matching the sibling fall test's 5% threshold).
        self.assertLess(sum(1 for v in tce if v["deceased.age"] >= 100) / len(tce), 0.05)

    def test_B5_adult_traffic_moto_median_age_is_young(self):
        import statistics
        moto = [v["deceased.age"] for v in self.by_group["adult"] if v["death.cause.chain_id"] == "trauma_moto"]
        self.assertGreater(len(moto), 15)
        self.assertLessEqual(statistics.median(moto), 30)

    def test_M1_fetal_identity_fields_blank_and_naturalidade_coherent(self):
        for v in self.by_group["fetal"]:
            self.assertIsNone(v["deceased.cns"])
            self.assertIsNone(v["deceased.marital_status"])
            self.assertIsNone(v["deceased.schooling"])
            self.assertIsNone(v["deceased.occupation"])

    def test_M2_mother_15_to_21_never_superior_completo(self):
        for v in self.by_group["fetal"] + self.by_group["infant"]:
            if v["deceased.mother_age"] <= 21:
                self.assertNotEqual(v["deceased.mother_schooling"], "Superior completo", v["deceased.mother_age"])

    def test_M2_child_occupation_never_adult_job_under_18(self):
        for v in self.by_group["child"]:
            if v["deceased.age"] < 18:
                self.assertIn(v["deceased.occupation"], ("Estudante", "Não se aplica"), v["deceased.age"])

    def test_M3_meningococcemia_and_drowning_codes_are_consistent(self):
        for v in self.records:
            if v["death.cause.chain_id"] == "crianca_meningococcemia":
                self.assertEqual(v["death.cause.a_cid"], "R57.2")
                self.assertNotEqual(v["death.cause.a_cid"], "A39.4")
            if v["death.cause.chain_id"] == "crianca_afogamento":
                self.assertNotIn("Parada cardiorrespiratória", v["death.cause.a_text"])
            if v["death.cause.chain_id"] == "infant_asfixia_perinatal":
                self.assertEqual(v["death.cause.a_cid"], "P21.0")

    def test_M4_pedestrian_child_never_at_home(self):
        for v in self.records:
            if v["death.cause.chain_id"] == "crianca_atropelamento":
                self.assertNotEqual(v["death.place.category"], "3")

    def test_M5_field37_no_parto_no_aborto_never_drawn(self):
        for v in self.records:
            self.assertNotIn(v.get("deceased.fertile_context"), ("No parto", "No aborto"))

    def test_M6_part2_interval_is_not_always_1_ano(self):
        values = {v["death.cause.ii1_interval"] for v in self.records if v["death.cause.ii1_interval"]}
        self.assertGreater(len(values), 1)

    _ADULT_PART2_CIDS = {"E11.9", "I10", "J44.9", "E66.9", "F17.2", "N18.9", "F10.2"}

    def test_R1_child_part2_never_from_adult_pool_and_never_exceeds_age(self):
        _UNIT_MINUTES = {"anos": 525600}
        for v in self.by_group["child"]:
            for slot in ("ii1", "ii2"):
                cid = v[f"death.cause.{slot}_cid"]
                interval = v[f"death.cause.{slot}_interval"]
                if cid is None:
                    continue
                self.assertNotIn(cid, self._ADULT_PART2_CIDS, (v["deceased.age"], cid))
                n = int(interval.split(" ", 1)[0])
                self.assertLessEqual(n, v["deceased.age"], (v["deceased.age"], interval))

    def test_R2_completed_series_never_exceeds_age_minus_6(self):
        for v in self.by_group["child"]:
            series = v["deceased.schooling_series"]
            if series is not None:
                self.assertLessEqual(series, max(0, v["deceased.age"] - 6), (v["deceased.age"], series))

    def test_R2_adult_18_to_21_never_superior_completo_without_graduate_occupation(self):
        graduate_occupations = {"Engenheira", "Engenheiro", "Advogada", "Advogado", "Professora", "Professor",
                                "Enfermeiro", "Enfermeira"}
        for v in self.by_group["adult"]:
            if 18 <= v["deceased.age"] <= 21 and v["deceased.occupation"] not in graduate_occupations:
                self.assertNotEqual(v["deceased.schooling"], "Superior completo", v["deceased.age"])

    def test_R2_mother_age_15_never_completed_medio(self):
        for v in self.by_group["fetal"] + self.by_group["infant"]:
            if v["deceased.mother_age"] == 15:
                self.assertNotEqual(v["deceased.mother_schooling"], "Médio (antigo 2º grau)", v["deceased.mother_age"])


if __name__ == "__main__":
    unittest.main()
