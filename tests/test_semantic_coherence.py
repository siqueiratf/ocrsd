"""Property tests for cross-field coherence of the semantic records: geography of the
event, residence and registry office, names of non-hospital health establishments, the
certifier role outside the DO, occupations inflected by sex, street variety, distinct
names across roles, and DNV obstetric coherence."""
import functools
import json
import tempfile
import unittest
from pathlib import Path

from src.semantic.data import (OCCUPATION_GENDER, OCCUPATION_OPPOSITE_FORM, OTHER_HEALTH_ESTABLISHMENT_NAMES,
                               REGION_BY_UF, LOCATION_PROFILES)
from src.semantic.generators import MOTHER_AGES, OCCUPATION_MIN_SCHOOLING, inflect_occupation
from src.semantic.plan import semantic_plan
from src.semantic.record import SemanticRecord, stored_value

N = 800
ROOT = Path(__file__).resolve().parents[1]


@functools.lru_cache(maxsize=None)
def _do_records(n=N, start=0):
    mapping = json.load(open("documents/do/mappings/mapeamento_do_v2.json", encoding="utf-8"))
    fields = [f for f in mapping["fields"] if not f.get("unsupported_reason")]
    plan = semantic_plan(fields, "declaracao_obito")
    records = []
    for seed in range(start, start + n):
        record = SemanticRecord(seed=seed, context="declaracao_obito")
        plan.apply(record)
        records.append(record._values)
    return records


@functools.lru_cache(maxsize=None)
def _cert_obito_records(n=N, start=0):
    mapping = json.load(open("documents/certidao_obito/mappings/" +
                             json.load(open("documents/certidao_obito/modelo.json"))["active_mapping"] + ".json",
                             encoding="utf-8"))
    fields = [f for f in mapping["fields"] if not f.get("unsupported_reason")]
    plan = semantic_plan(fields, "certidao_obito")
    records = []
    for seed in range(start, start + n):
        record = SemanticRecord(seed=seed, context="certidao_obito")
        plan.apply(record)
        records.append(record._values)
    return records


@functools.lru_cache(maxsize=None)
def _dnv_records(n=N, start=0):
    mapping = json.load(open("documents/dnv/mappings/" +
                             json.load(open("documents/dnv/modelo.json"))["active_mapping"] + ".json",
                             encoding="utf-8"))
    fields = [f for f in mapping["fields"] if not f.get("unsupported_reason")]
    plan = semantic_plan(fields, "dnv")
    records = []
    for seed in range(start, start + n):
        record = SemanticRecord(seed=seed, context="dnv")
        plan.apply(record)
        records.append(record._values)
    return records


@functools.lru_cache(maxsize=None)
def _cnj_records(n=N, start=0):
    mapping = json.load(open("documents/certidao_nascimento_cnj/mappings/" +
                             json.load(open("documents/certidao_nascimento_cnj/modelo.json"))["active_mapping"] + ".json",
                             encoding="utf-8"))
    fields = [f for f in mapping["fields"] if not f.get("unsupported_reason")]
    plan = semantic_plan(fields, "certidao_nascimento_cnj")
    records = []
    for seed in range(n):
        record = SemanticRecord(seed=seed, context="certidao_nascimento_cnj")
        plan.apply(record)
        records.append(record._values)
    return records


class GeographyTests(unittest.TestCase):
    """Item 1: Lei 6.015/73 art. 77/50 — registration at the place of the event or of
    residence; cross-state events should be the exception (~5%), biased toward
    neighbouring states, not spread uniformly over the whole country."""

    @classmethod
    def setUpClass(cls):
        cls.do_records = _do_records()

    def test_residence_uf_rarely_crosses_state_and_favors_same_region(self):
        cross = same_region = 0
        for v in self.do_records:
            death_uf, residence_uf = v["death.place.uf"], v["deceased.residence.uf"]
            if death_uf != residence_uf:
                cross += 1
                if REGION_BY_UF[death_uf] == REGION_BY_UF[residence_uf]:
                    same_region += 1
        rate = cross / len(self.do_records)
        self.assertLess(rate, 0.12, f"residence UF != death UF in {rate:.1%} (real-world ~5%)")
        if cross:
            self.assertGreater(same_region / cross, 0.4, "cross-state residence should favor the same region")

    def test_cartorio_is_almost_always_the_death_or_residence_uf(self):
        # R-a: Lei 6.015/73 art. 77 allows ONLY the place of death or of residence — 0%
        # unrelated, not merely "almost always".
        unrelated = 0
        for v in self.do_records:
            ufs = {v["death.place.uf"], v["deceased.residence.uf"]}
            if v["death_registration.office.uf"] not in ufs:
                unrelated += 1
        self.assertEqual(unrelated, 0, "cartório UF unrelated to death/residence (Lei 6.015/73 art. 77 allows only those two)")

    def test_birth_registration_office_favors_birth_or_mother_residence(self):
        # R-a: Lei 6.015/73 art. 50 allows ONLY the place of birth or of the mother's
        # residence — 0% unrelated.
        records = _dnv_records(600)
        unrelated = 0
        for v in records:
            ufs = {v["birth.place.uf"], v["mother.residence.uf"]}
            if v["registration.office.uf"] not in ufs:
                unrelated += 1
        self.assertEqual(unrelated, 0, "birth cartório UF unrelated to birth/residence (Lei 6.015/73 art. 50 allows only those two)")

    def test_mother_residence_rarely_crosses_state_from_birth_place(self):
        records = _dnv_records(600)
        cross = sum(1 for v in records if v["birth.place.uf"] != v["mother.residence.uf"])
        rate = cross / len(records)
        self.assertLess(rate, 0.12, f"mother residence UF != birth UF in {rate:.1%}")


class EstablishmentNameTests(unittest.TestCase):
    """Item 2: field-20 category 2 ("outro estabelecimento de saúde") must never be a
    "Hospital ..."; it should be a UPA/UBS/policlínica/pronto atendimento."""

    def test_category_2_never_named_hospital(self):
        records = _do_records()
        seen_category_2 = 0
        for v in records:
            if v["death.place.category"] == "2":
                seen_category_2 += 1
                name = v["death.place.establishment.name"]
                self.assertIn(name, OTHER_HEALTH_ESTABLISHMENT_NAMES, f"category 2 named {name!r}")
                self.assertNotIn("Hospital", name)
        self.assertGreater(seen_category_2, 0)


class CertifierRoleTests(unittest.TestCase):
    """Item 3: outside the DO, the certifier's role must always be "1" — no IML/SVO
    (hence no necropsy "Não" context) on a document that isn't the DO."""

    def test_certidao_obito_role_always_1(self):
        records = _cert_obito_records(200)
        for v in records:
            self.assertEqual(v["certifier.role"], "1")
            self.assertIsNone(v["certifier.iml_svo_municipality"])


class OccupationSexTests(unittest.TestCase):
    """Item 4: occupation titles agree with the person's sex."""

    def test_inflect_occupation_matches_gender_dict(self):
        for name, gender in OCCUPATION_GENDER.items():
            self.assertEqual(inflect_occupation(name, gender), name)
            opposite = OCCUPATION_OPPOSITE_FORM[name]
            other_sex = "M" if gender == "F" else "F"
            self.assertEqual(inflect_occupation(name, other_sex), opposite)

    def test_deceased_occupation_agrees_with_sex(self):
        records = _do_records()
        bad = []
        for v in records:
            occ, sex = v["deceased.occupation"], v["deceased.sex"]
            if occ in OCCUPATION_GENDER and OCCUPATION_GENDER[occ] != sex:
                bad.append((occ, sex))
        self.assertEqual(bad, [])

    def test_mother_occupation_agrees_with_female_sex(self):
        records = _dnv_records()
        bad = [v["mother.occupation"] for v in records
              if v["mother.occupation"] in OCCUPATION_GENDER and OCCUPATION_GENDER[v["mother.occupation"]] != "F"]
        self.assertEqual(bad, [])

    def test_occupation_cbo_still_correct_after_inflection(self):
        from src.semantic.constraints import CONSTRAINTS, violations
        records = _do_records()
        names = {c.name for c in CONSTRAINTS}
        self.assertIn("ocupacao_do_falecido_e_cbo", names)
        for v in records:
            messages = [m for m, _ in violations(v) if "ocupacao_do_falecido_e_cbo" in m or "CBO" in m]
            self.assertEqual(messages, [], v.get("deceased.occupation"))

    def test_tecnico_em_informatica_and_auxiliar_administrativo_have_feminine_forms(self):
        # R-b: these two were missing their feminine form (261 of 2000 DNV mothers affected).
        self.assertEqual(inflect_occupation("Técnico em informática", "F"), "Técnica em informática")
        self.assertEqual(inflect_occupation("Técnica em informática", "M"), "Técnico em informática")
        self.assertEqual(inflect_occupation("Auxiliar administrativo", "F"), "Auxiliar administrativa")
        self.assertEqual(inflect_occupation("Auxiliar administrativa", "M"), "Auxiliar administrativo")


class AiOccupationAgreesWithSexTests(unittest.TestCase):
    """R-c: an occupation value required by the AI (always a canonical form) must still be
    inflected to agree with the record's sex in the final, stored value."""

    def test_ai_required_mother_occupation_is_inflected_for_female_sex(self):
        record = SemanticRecord(seed=1, context="dnv")
        # assisted.py widens the AI's single canonical answer to both grammatical forms
        # (OCCUPATION_SLOT_KEYS) in one require() call, so the generator's inflected value
        # — not necessarily the literal AI answer — can still satisfy it.
        record.require("mother.occupation", ("Enfermeiro", "Enfermeira"), "valor proposto pela IA (ambos os gêneros)")
        values = record.resolve(["mother.occupation", "mother.occupation_cbo"])
        self.assertEqual(values["mother.occupation"], "Enfermeira")

    def test_ai_required_deceased_occupation_is_inflected_for_male_sex(self):
        record = SemanticRecord(seed=2, context="declaracao_obito")
        record.require("deceased.sex", ("M",), "valor proposto pela IA")
        record.require("deceased.age", (40,), "valor proposto pela IA")
        record.require("deceased.occupation", ("Professora", "Professor"), "valor proposto pela IA (ambos os gêneros)")
        values = record.resolve(["deceased.sex", "deceased.age", "deceased.occupation"])
        self.assertEqual(values["deceased.occupation"], "Professor")


class StreetVarietyTests(unittest.TestCase):
    """Item 5: each municipality offers several streets, so repeated "another address"
    draws are not always the same street."""

    def test_every_profile_has_several_distinct_streets(self):
        for profile in LOCATION_PROFILES:
            streets = profile["streets"]
            self.assertGreaterEqual(len(streets), 5, profile["city"])
            self.assertEqual(len({s for s, _, _ in streets}), len(streets), profile["city"])
            self.assertEqual(len({c for _, c, _ in streets}), len(streets), profile["city"])

    def test_residence_addresses_use_more_than_the_profile_primary_street(self):
        records = _do_records()
        streets = {v["deceased.residence.street"] for v in records}
        self.assertGreater(len(streets), 12, "residence addresses should use many distinct streets")

    def test_occurrence_addresses_are_not_limited_to_twelve_streets(self):
        records = _do_records()
        streets = {v["deceased.occurrence.address"] for v in records if v["deceased.occurrence.address"]}
        self.assertGreater(len(streets), 12, f"only {len(streets)} distinct occurrence streets (R4 limitation)")


class DistinctNameTests(unittest.TestCase):
    """Item 6: no two distinct family roles share a full name by chance."""

    def test_father_name_never_equals_mother_name(self):
        for seed in range(N):
            record = SemanticRecord(seed=seed, context="dnv")
            self.assertNotEqual(record.value("father.name"), record.value("mother.name"))

    def test_newborn_name_never_equals_a_parent_name(self):
        for seed in range(N):
            record = SemanticRecord(seed=seed, context="dnv")
            newborn = record.value("newborn.name")
            self.assertNotEqual(newborn, record.value("mother.name"))
            self.assertNotEqual(newborn, record.value("father.name"))

    def test_certidao_nascimento_cnj_parent_names_never_collide(self):
        records = _cnj_records(200)
        collisions = sum(1 for v in records if v["parent1.name"].strip().lower() == v["parent2.name"].strip().lower())
        self.assertEqual(collisions, 0)
        collisions_with_newborn = sum(
            1 for v in records
            if v["newborn.name"].strip().lower() in {v["parent1.name"].strip().lower(), v["parent2.name"].strip().lower()})
        self.assertEqual(collisions_with_newborn, 0)


class DnvCoherenceTests(unittest.TestCase):
    """Item 7: DNV mother/newborn relations not previously covered by a property test."""

    @classmethod
    def setUpClass(cls):
        cls.records = _dnv_records()

    def test_apgar_scores_are_valid_and_non_decreasing(self):
        for v in self.records:
            first, fifth = v["newborn.apgar_1min"], v["newborn.apgar_5min"]
            self.assertTrue(0 <= first <= 10)
            self.assertTrue(0 <= fifth <= 10)
            self.assertGreaterEqual(fifth, first)

    def test_birth_weight_tracks_gestational_age(self):
        preterm = [v["newborn.weight_g"] for v in self.records if v["pregnancy.weeks"] < 34]
        term = [v["newborn.weight_g"] for v in self.records if 38 <= v["pregnancy.weeks"] <= 41]
        if preterm and term:
            self.assertLess(sum(preterm) / len(preterm), sum(term) / len(term),
                            "preterm births should be lighter on average than term births")

    def test_mother_obstetric_history_consistent_with_age(self):
        for v in self.records:
            age = v["mother.age"]
            previous = v["mother.obstetric.previous_pregnancies"]
            # A mother cannot plausibly have had more pregnancies than roughly one every two
            # years of fertile life (from 13): gestational_history() already caps the draw this
            # way; this test locks the invariant down with a property check over many seeds.
            self.assertLessEqual(previous, max(0, (age - 13) // 2 + 1) + 3)

    def test_prenatal_visits_consistent_with_gestation_length(self):
        for v in self.records:
            weeks, visits = v["pregnancy.weeks"], v["pregnancy.prenatal_visits"]
            # A full-term pregnancy with 0 prenatal visits is rare but possible (no modeled
            # upper cap is violated); the real invariant is that visits never wildly exceed
            # what's physically possible in the gestation window (roughly one every 2 weeks).
            self.assertLessEqual(visits, weeks // 2 + 3, (weeks, visits))

    def test_twin_pregnancy_fields_agree(self):
        # The active DNV mapping does not surface "twin.name_matricula" as a field, so the
        # key may be absent from the resolved record; when present it must agree with twins.
        for v in self.records:
            if v["birth.twins"]:
                self.assertEqual(v["birth.pregnancy_type"], "Dupla")
                if "twin.name_matricula" in v:
                    self.assertIsNotNone(v["twin.name_matricula"])
            else:
                self.assertEqual(v["birth.pregnancy_type"], "Única")
                if "twin.name_matricula" in v:
                    self.assertIsNone(v["twin.name_matricula"])

    def test_mother_residence_rarely_crosses_state_from_birth(self):
        cross = sum(1 for v in self.records if v["birth.place.uf"] != v["mother.residence.uf"])
        self.assertLess(cross / len(self.records), 0.12)


class MotherOccupationSchoolingAgeTests(unittest.TestCase):
    """R-d: the DNV mother's occupation, schooling and age must agree — no graduate
    occupation under ~22, no technical occupation below ~16 (which also collapses the
    occupation to "Estudante"), and schooling plausible for age. Also m-3: age in [13, 49]."""

    @classmethod
    def setUpClass(cls):
        cls.records = _dnv_records()

    def test_mother_age_is_within_a_realistic_range(self):
        self.assertEqual(MOTHER_AGES.start, 13)
        self.assertEqual(MOTHER_AGES.stop, 50)
        for v in self.records:
            self.assertTrue(13 <= v["mother.age"] <= 49, v["mother.age"])

    def test_graduate_occupation_never_appears_under_minimum_age(self):
        bad = []
        for v in self.records:
            minimum = OCCUPATION_MIN_SCHOOLING.get(v["mother.occupation"]) or OCCUPATION_MIN_SCHOOLING.get(
                OCCUPATION_OPPOSITE_FORM.get(v["mother.occupation"], ""))
            if minimum == "Superior completo" and v["mother.age"] < 22:
                bad.append((v["mother.occupation"], v["mother.age"]))
            if minimum == "Médio (antigo 2º grau)" and v["mother.age"] < 16:
                bad.append((v["mother.occupation"], v["mother.age"]))
        self.assertEqual(bad, [])

    def test_mother_under_sixteen_is_always_estudante(self):
        bad = [(v["mother.age"], v["mother.occupation"]) for v in self.records
              if v["mother.age"] < 16 and v["mother.occupation"] != "Estudante"]
        self.assertEqual(bad, [])

    def test_occupation_implies_at_least_its_minimum_schooling(self):
        from src.semantic.generators import SCHOOLING_LEVELS
        order = {level: index for index, level in enumerate(SCHOOLING_LEVELS)}
        bad = []
        for v in self.records:
            minimum = OCCUPATION_MIN_SCHOOLING.get(v["mother.occupation"]) or OCCUPATION_MIN_SCHOOLING.get(
                OCCUPATION_OPPOSITE_FORM.get(v["mother.occupation"], ""))
            if minimum and order[v["mother.schooling"]] < order[minimum]:
                bad.append((v["mother.occupation"], v["mother.schooling"]))
        self.assertEqual(bad, [])


class AiOccupationAgreesWithSexTests(unittest.TestCase):
    """R-c: an occupation value required by the AI (always a canonical form) must still be
    inflected to agree with the record's sex in the final, stored value."""

    def test_ai_required_mother_occupation_is_inflected_for_female_sex(self):
        record = SemanticRecord(seed=1, context="dnv")
        # assisted.py widens the AI's single canonical answer to both grammatical forms
        # (OCCUPATION_SLOT_KEYS) in one require() call, so the generator's inflected value
        # — not necessarily the literal AI answer — can still satisfy it.
        record.require("mother.occupation", ("Enfermeiro", "Enfermeira"), "valor proposto pela IA (ambos os gêneros)")
        values = record.resolve(["mother.occupation", "mother.occupation_cbo"])
        self.assertEqual(values["mother.occupation"], "Enfermeira")

    def test_ai_required_deceased_occupation_is_inflected_for_male_sex(self):
        record = SemanticRecord(seed=2, context="declaracao_obito")
        record.require("deceased.sex", ("M",), "valor proposto pela IA")
        record.require("deceased.age", (40,), "valor proposto pela IA")
        record.require("deceased.occupation", ("Professora", "Professor"), "valor proposto pela IA (ambos os gêneros)")
        values = record.resolve(["deceased.sex", "deceased.age", "deceased.occupation"])
        self.assertEqual(values["deceased.occupation"], "Professor")

    def test_fake_ai_run_for_dnv_with_required_occupation_agrees_with_mother_sex(self):
        # End-to-end through assisted.py's own require()-widening (OCCUPATION_SLOT_KEYS),
        # with a stub provider standing in for the LLM — not a direct SemanticRecord call.
        from src.semantic.assisted import AssistedRecords
        from src.synthetic_dataset import generate_dataset

        class FakeAssistant:
            provider, model = "openai", "fake"

            def complete_json(self, system, payload, *, temperature, seed):
                values = {}
                for campo in payload["campos"]:
                    key = campo["chave"]
                    if key == "mother.occupation":
                        values[key] = "Enfermeiro"  # canonical, masculine, requires inflection for the mother (F)
                    elif "opcoes" in campo:
                        values[key] = campo["opcoes"][0]
                    elif "intervalo" in campo:
                        values[key] = sum(campo["intervalo"]) // 2
                    else:
                        values[key] = "Maria Clara Souza"
                return {"valores": values}, {"model": "fake", "seconds": 0.01, "prompt_tokens": 1, "completion_tokens": 1}

        mapping_path = ROOT / ("documents/dnv/mappings/" + json.load(open(ROOT / "documents/dnv/modelo.json"))["active_mapping"] + ".json")
        definition = json.loads(mapping_path.read_text(encoding="utf-8"))
        selected = {field["field_id"] for field in definition["fields"] if not field.get("unsupported_reason")}
        factory = AssistedRecords(FakeAssistant(), fields=definition["fields"], dpi=300)
        with tempfile.TemporaryDirectory() as directory:
            generated = generate_dataset(ROOT / "documents/dnv/template.pdf", mapping_path, selected, Path(directory), 1,
                                         seed=123, handwriting=False, text_handwriting=False, record_factory=factory)
            truth = json.loads(generated[0].json_path.read_text(encoding="utf-8"))
            self.assertEqual(truth["semantic_provenance"]["mode"], "ai")
            self.assertEqual(truth["semantic_record"]["mother.occupation"], "Enfermeira")


class BairroPerStreetTests(unittest.TestCase):
    """m-1: each street carries its own real/plausible bairro, instead of every address in
    a municipality sharing the profile's single default bairro."""

    def test_several_profiles_offer_more_than_one_bairro(self):
        # Not every profile's streets must span multiple bairros (some, like Rio's chosen
        # streets, are genuinely all "Centro") — but most should, unlike the pre-fix state
        # where every street in every profile echoed the profile's single default bairro.
        multi_bairro_profiles = sum(1 for profile in LOCATION_PROFILES if len({entry[2] for entry in profile["streets"]}) > 1)
        self.assertGreater(multi_bairro_profiles, len(LOCATION_PROFILES) // 2)

    def test_residence_bairro_matches_its_own_street_not_always_the_profile_default(self):
        records = _dnv_records(300)
        mismatched_default = 0
        for v in records:
            profile = next(p for p in LOCATION_PROFILES if p["ibge"] == v["mother.residence.ibge"])
            street_bairro = next(entry[2] for entry in profile["streets"] if entry[0] == v["mother.residence.street"])
            self.assertEqual(v["mother.residence.neighborhood"], street_bairro)
            if street_bairro != profile["neighborhood"]:
                mismatched_default += 1
        self.assertGreater(mismatched_default, 0, "no residence used a non-default bairro across 300 records")


class AnomalyAndCboTests(unittest.TestCase):
    """Item 8: DNV congenital-anomaly block (field 6 status + field 41 description/ICD-10
    code) and CBO-2002 codes for every working occupation (DNV mother, DO deceased, DO
    deceased's mother)."""

    @classmethod
    def setUpClass(cls):
        cls.dnv = _dnv_records(2000)
        cls.do = _do_records(2000)

    def test_anomaly_status_distribution_is_realistic(self):
        from collections import Counter
        counts = Counter(v["newborn.anomaly_status"] for v in self.dnv)
        total = len(self.dnv)
        self.assertLess(counts["Sim"] / total, 0.04)
        self.assertGreater(counts["Não"] / total, 0.90)
        self.assertLess(counts.get("Ignorado", 0) / total, 0.04)

    def test_anomaly_block_is_coherent(self):
        import re
        from src.semantic.data import ANOMALY_CODES
        code_re = re.compile(r"\(([A-Z]\d{2}(?:\.\d)?)\)$")
        sim_count = 0
        for v in self.dnv:
            status, description = v["newborn.anomaly_status"], v["newborn.anomaly_description"]
            if status == "Sim":
                sim_count += 1
                self.assertIsNotNone(description)
                match = code_re.search(description)
                self.assertIsNotNone(match, description)
                self.assertIn(match.group(1), ANOMALY_CODES)
            else:
                self.assertIsNone(description)
        self.assertGreater(sim_count, 0, "no “Sim” case drawn across 2000 seeds")

    def test_hypospadia_only_affects_male_newborns(self):
        for v in self.dnv:
            description, sex = v["newborn.anomaly_description"], v["newborn.sex"]
            if description and "Q54.9" in description:
                self.assertEqual(sex, "M")

    def test_cbo_present_and_well_formed_for_every_working_occupation(self):
        from src.semantic.data import occupation_cbo_for
        samples = [(v["mother.occupation"], v["mother.occupation_cbo"]) for v in self.dnv]
        samples += [(v["deceased.occupation"], v["deceased.occupation_cbo"]) for v in self.do]
        samples += [(v["deceased.mother_occupation"], v["deceased.mother_occupation_cbo"]) for v in self.do
                    if v.get("deceased.mother_occupation")]
        non_cbo = {"Estudante", "Não se aplica", None}
        seen_working = set()
        for occupation, cbo in samples:
            self.assertEqual(cbo, occupation_cbo_for(occupation), occupation)
            if occupation in non_cbo:
                self.assertIsNone(cbo, occupation)
            else:
                self.assertIsNotNone(cbo, occupation)
                self.assertRegex(cbo, r"^\d{6}$", occupation)
                seen_working.add(occupation)
        self.assertGreater(len(seen_working), 5, "too few distinct working occupations sampled")

    def test_cbo_stable_under_sex_inflection(self):
        from src.semantic.data import OCCUPATION_OPPOSITE_FORM, occupation_cbo_for
        for occupation in OCCUPATION_OPPOSITE_FORM:
            opposite = OCCUPATION_OPPOSITE_FORM[occupation]
            self.assertEqual(occupation_cbo_for(occupation), occupation_cbo_for(opposite), occupation)


if __name__ == "__main__":
    unittest.main()
