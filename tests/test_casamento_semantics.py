import json
import tempfile
import unittest
from datetime import date
from pathlib import Path

import fitz

from src.semantic.assisted import AssistedRecords, SLOTS
from src.semantic.constraints import completed_years, violations
from src.semantic.record import SemanticRecord, stored_value
from src.semantic.registry import ENTITY_KEYS, REGISTRY
from src.semantic.resolve import document_context
from src.semantic.plan import semantic_plan
from src.synthetic_dataset import generate_dataset

ROOT = Path(__file__).resolve().parents[1]
CASAMENTO_CNJ_MAPPING = ROOT / "documents/certidao_casamento_cnj/mappings/mapeamento_certidao_casamento_cnj_v1.json"
CASAMENTO_CNJ_TEMPLATE = ROOT / "documents/certidao_casamento_cnj/template.pdf"
CASAMENTO_MAPPING = ROOT / "documents/certidao_casamento/mappings/mapeamento_certidao_casamento_v1.json"
OBITO_MAPPING = ROOT / "documents/certidao_obito/mappings/mapeamento_certidao_obito_v1.json"
NASCIMENTO_CNJ_MAPPING = ROOT / "documents/certidao_nascimento_cnj/mappings/mapeamento_certidao_nascimento_cnj_v1.json"

MARRIAGE_KEYS = (ENTITY_KEYS["spouse1"] + ENTITY_KEYS["spouse2"] + ENTITY_KEYS["marriage"]
                 + ENTITY_KEYS["marriage_registration"])
OFFICE_CERT_KEYS = ENTITY_KEYS["office"] + ENTITY_KEYS["certificate"]


class FakeAssistant:
    provider = "openai"
    model = "fake"

    def __init__(self):
        self.calls = []

    def complete_json(self, system, payload, *, temperature, seed):
        self.calls.append(payload)
        values = {}
        for campo in payload["campos"]:
            key = campo["chave"]
            if "opcoes" in campo:
                values[key] = campo["opcoes"][0]
            elif "intervalo" in campo:
                values[key] = sum(campo["intervalo"]) // 2
            else:
                values[key] = "Maria Clara Souza"
        return {"valores": values}, {"model": "fake", "seconds": 0.01, "prompt_tokens": 1, "completion_tokens": 1}


class RegistryLoadTests(unittest.TestCase):
    def test_new_marriage_and_office_keys_are_registered(self):
        expected = {
            "spouse1.name", "spouse1.name_after", "spouse1.cpf", "spouse1.birth_date", "spouse1.age",
            "spouse1.nationality", "spouse1.marital_status_before", "spouse1.naturalidade.municipality",
            "spouse1.naturalidade.uf", "spouse1.parents_names", "marriage.date", "marriage.property_regime",
            "marriage.celebrant.name", "marriage_registration.date", "marriage_registration.matricula",
            "office.cns", "office.official_name", "office.city_uf", "office.address",
            "certificate.issue_date", "certificate.clerk_name", "certificate.clerk_role", "certificate.issue_summary",
        }
        self.assertTrue(expected <= set(REGISTRY))
        for key in expected:
            self.assertIn("certidao_casamento", REGISTRY[key].contexts)

    def test_death_and_birth_extras_are_registered(self):
        for key in ("deceased.spouse_name", "deceased.marital_status", "deceased.naturalidade.municipality",
                   "deceased.naturalidade.uf", "deceased.parents_names", "burial.place", "burial.municipality",
                   "burial.uf", "death_registration.declarant_name", "death_registration.matricula"):
            self.assertIn("declaracao_obito", REGISTRY[key].contexts, key)
        for key in ("parent1.name", "parent2.name", "parent1.naturalidade.municipality", "parent1.naturalidade.uf",
                   "parent1.grandparents_names", "newborn.naturalidade.uf"):
            self.assertIn("certidao_nascimento", REGISTRY[key].contexts, key)


class ContextMappingTests(unittest.TestCase):
    def test_casamento_documents_map_to_the_new_context(self):
        for name in ("certidao_casamento", "certidao_casamento_cnj", "Certidão de Casamento"):
            self.assertEqual(document_context(name), "certidao_casamento")

    def test_existing_contexts_are_unaffected(self):
        self.assertEqual(document_context("dnv"), "dnv")
        self.assertEqual(document_context("certidao_nascimento"), "certidao_nascimento")
        self.assertEqual(document_context("certidao_nascimento_cnj"), "certidao_nascimento")
        self.assertEqual(document_context("certidao_obito"), "declaracao_obito")
        self.assertEqual(document_context("declaracao_obito"), "declaracao_obito")
        self.assertEqual(document_context("do"), "declaracao_obito")


class DeterministicCoherenceTests(unittest.TestCase):
    def test_spouse_ages_and_property_regime(self):
        forced_regime = 0
        for seed in range(400):
            record = SemanticRecord(seed, "certidao_casamento")
            values = record.resolve(MARRIAGE_KEYS)
            self.assertEqual(violations(values), [])
            for number in (1, 2):
                age = completed_years(values[f"spouse{number}.birth_date"], values["marriage.date"])
                self.assertEqual(age, values[f"spouse{number}.age"])
                self.assertGreaterEqual(age, 16)
            if max(values["spouse1.age"], values["spouse2.age"]) >= 70:
                forced_regime += 1
                self.assertEqual(values["marriage.property_regime"], "SEPARAÇÃO OBRIGATÓRIA DE BENS")
            else:
                self.assertNotEqual(values["marriage.property_regime"], "SEPARAÇÃO OBRIGATÓRIA DE BENS")
        self.assertGreater(forced_regime, 0)

    def test_name_after_is_derived_from_both_names(self):
        for seed in range(50):
            values = SemanticRecord(seed, "certidao_casamento").resolve(MARRIAGE_KEYS)
            for number, other in ((1, 2), (2, 1)):
                own, other_name = values[f"spouse{number}.name"], values[f"spouse{other}.name"]
                after = values[f"spouse{number}.name_after"]
                self.assertEqual(after.split()[0], own.split()[0])
                self.assertTrue(set(after.split()[1:]) <= set(own.split()[1:]) | {other_name.split()[-1]})

    def test_registration_is_not_before_the_celebration(self):
        for seed in range(100):
            values = SemanticRecord(seed, "certidao_casamento").resolve(MARRIAGE_KEYS)
            self.assertGreaterEqual(values["marriage_registration.date"], values["marriage.date"])
            matricula = values["marriage_registration.matricula"]
            self.assertEqual(len(matricula), 32)
            self.assertTrue(matricula.isdigit())
            self.assertEqual(matricula[14], "2")
            self.assertEqual(matricula[10:14], f"{values['marriage_registration.date'].year:04d}")

    def test_death_registration_matricula_uses_livro_4(self):
        for seed in range(50):
            values = SemanticRecord(seed, "declaracao_obito").resolve(
                ENTITY_KEYS["death_registration"] + ENTITY_KEYS["burial"] + ENTITY_KEYS["death"])
            matricula = values["death_registration.matricula"]
            self.assertEqual(matricula[14], "4")
            self.assertEqual(matricula[10:14], f"{values['death_registration.date'].year:04d}")

    def test_burial_municipality_and_uf_match_a_catalog_profile(self):
        from src.semantic.data import LOCATION_PROFILES
        for seed in range(80):
            values = SemanticRecord(seed, "declaracao_obito").resolve(ENTITY_KEYS["burial"] + ENTITY_KEYS["death"])
            self.assertTrue(any(profile["city"] == values["burial.municipality"] and profile["uf"] == values["burial.uf"]
                               for profile in LOCATION_PROFILES))

    def test_office_and_certificate_issue_date_ordering_for_all_three_contexts(self):
        cases = (
            ("certidao_nascimento", "registration.date", ENTITY_KEYS["registration"]),
            ("declaracao_obito", "death_registration.date", ENTITY_KEYS["death_registration"]),
            ("certidao_casamento", "marriage_registration.date", MARRIAGE_KEYS),
        )
        for context, reg_key, extra_keys in cases:
            for seed in range(40):
                values = SemanticRecord(seed, context).resolve(list(extra_keys) + list(OFFICE_CERT_KEYS))
                self.assertGreaterEqual(values["certificate.issue_date"], values[reg_key], context)
                self.assertLessEqual(values["certificate.issue_date"], date(2025, 12, 31))
                self.assertTrue(values["office.address"].endswith(values["office.city_uf"]))

    def test_parent1_and_parent2_are_never_the_same_person(self):
        keys = ENTITY_KEYS["parent1"] + ENTITY_KEYS["parent2"] + ("mother.name", "father.name")
        aliases_mother = 0
        for seed in range(200):
            values = SemanticRecord(seed, "certidao_nascimento").resolve(keys)
            self.assertNotEqual(values["parent1.name"], values["parent2.name"])
            self.assertIn(values["parent1.name"], {values["mother.name"], values["father.name"]})
            self.assertIn(values["parent2.name"], {values["mother.name"], values["father.name"]})
            aliases_mother += values["parent1.name"] == values["mother.name"]
        self.assertTrue(50 < aliases_mother < 150)


class MappingBindingTests(unittest.TestCase):
    def test_all_four_mappings_resolve_without_issues(self):
        for path in (CASAMENTO_CNJ_MAPPING, CASAMENTO_MAPPING, OBITO_MAPPING, NASCIMENTO_CNJ_MAPPING):
            definition = json.loads(path.read_text(encoding="utf-8"))
            context = document_context(definition["template_id"])
            plan = semantic_plan(definition["fields"], context)
            self.assertEqual(plan.issues, (), path)
            self.assertFalse(any(decision.invalid for decision in plan.decisions.values()), path)

    def test_three_casamento_cnj_documents_are_coherent(self):
        definition = json.loads(CASAMENTO_CNJ_MAPPING.read_text(encoding="utf-8"))
        selected = {field["field_id"] for field in definition["fields"]}
        with tempfile.TemporaryDirectory() as directory:
            generated = generate_dataset(CASAMENTO_CNJ_TEMPLATE, CASAMENTO_CNJ_MAPPING, selected, Path(directory), 3,
                                         seed=4242, handwriting=False, text_handwriting=False)
            for document in generated:
                with fitz.open(document.pdf_path) as pdf:
                    self.assertEqual(len(pdf), 1)
                truth = json.loads(document.json_path.read_text(encoding="utf-8"))
                record = truth["semantic_record"]
                self.assertLessEqual(record["marriage.date"], record["marriage_registration.date"])
                self.assertLessEqual(record["marriage_registration.date"], record["certificate.issue_date"])
                self.assertEqual(record["office.city_uf"], f"{record['office.city_uf'].split(' - ')[0]} - "
                                                           f"{record['office.city_uf'].split(' - ')[1]}")


class AiSlotTests(unittest.TestCase):
    def test_slots_for_certidao_casamento_cover_the_main_fields(self):
        keys = {slot.key for slot in SLOTS["certidao_casamento"]}
        self.assertTrue({"spouse1.name", "spouse2.name", "spouse1.age", "spouse2.age", "spouse1.nationality",
                         "spouse1.marital_status_before", "spouse1.naturalidade.municipality",
                         "marriage.property_regime", "marriage.celebrant.name"} <= keys)

    def test_slots_for_certidao_nascimento_are_defined(self):
        keys = {slot.key for slot in SLOTS["certidao_nascimento"]}
        self.assertTrue({"newborn.name", "mother.name", "father.name", "newborn.naturalidade"} <= keys)

    def test_fake_ai_run_for_casamento_mapping_satisfies_constraints(self):
        definition = json.loads(CASAMENTO_CNJ_MAPPING.read_text(encoding="utf-8"))
        selected = {field["field_id"] for field in definition["fields"]}
        assistant = FakeAssistant()
        factory = AssistedRecords(assistant, fields=definition["fields"], dpi=300)
        with tempfile.TemporaryDirectory() as directory:
            generated = generate_dataset(CASAMENTO_CNJ_TEMPLATE, CASAMENTO_CNJ_MAPPING, selected, Path(directory), 1,
                                         seed=99, handwriting=False, text_handwriting=False, record_factory=factory)
            truth = json.loads(generated[0].json_path.read_text(encoding="utf-8"))
            self.assertEqual(truth["semantic_provenance"]["mode"], "ai")
            raw = truth["semantic_record"]
            record = {key: stored_value(key, value) if value is not None else None for key, value in raw.items()
                     if key in REGISTRY}
            self.assertEqual(violations(record), [])
            self.assertGreaterEqual(record["marriage_registration.date"], record["marriage.date"])


if __name__ == "__main__":
    unittest.main()
