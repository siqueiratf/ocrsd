"""Coherence of the semantic records of the DNV and of the certidões: matrícula check digits and
office CNS, issue dates, couples, second parent block, DNV birth scenarios and stillbirths."""
import functools
import json
import unittest
from collections import Counter
from datetime import date
from pathlib import Path

from src.semantic.plan import semantic_plan
from src.semantic.present import matricula_check_digits
from src.semantic.record import SemanticRecord
from src.semantic.resolve import document_context

ROOT = Path(__file__).resolve().parents[1]
N = 400


def _mapping(document: str) -> dict:
    folder = ROOT / "documents" / document
    active = json.loads((folder / "modelo.json").read_text(encoding="utf-8"))["active_mapping"]
    return json.loads((folder / "mappings" / f"{active}.json").read_text(encoding="utf-8"))


@functools.lru_cache(maxsize=None)
def records(document: str, n: int = N) -> list[tuple[dict, SemanticRecord]]:
    context = document_context(document)
    fields = [field for field in _mapping(document)["fields"]
              if not field.get("unsupported_reason") and not str(field.get("semantic", "")).startswith("custom.")]
    plan = semantic_plan(fields, context)
    found = []
    for seed in range(n):
        record = SemanticRecord(seed, context)
        plan.apply(record)
        found.append((record._values, record))
    return found


CERTIDOES = ("certidao_nascimento_cnj", "certidao_obito", "certidao_casamento", "certidao_casamento_cnj")
MATRICULA_KEYS = ("registration.matricula", "death_registration.matricula", "marriage_registration.matricula")


class MatriculaTests(unittest.TestCase):
    def test_each_single_digit_change_changes_the_check_digits(self):
        body = "104539015520131000120210000123"
        for index in range(30):
            changed = body[:index] + str((int(body[index]) + 1) % 10) + body[index + 1:]
            self.assertNotEqual(matricula_check_digits(body), matricula_check_digits(changed), index)

    def test_matricula_has_valid_check_digits_and_the_office_cns(self):
        for document in CERTIDOES:
            for values, _ in records(document):
                key = next(key for key in MATRICULA_KEYS if values.get(key))
                code = values[key]
                self.assertEqual(code[30:], matricula_check_digits(code[:30]), (document, code))
                if "office.cns" in values:
                    self.assertEqual(code[:6], values["office.cns"], document)

    def test_issue_date_is_in_the_matricula_era_and_after_the_registration(self):
        for document in ("certidao_nascimento_cnj", "certidao_obito", "certidao_casamento_cnj"):
            for values, _ in records(document):
                issued = values["certificate.issue_date"]
                self.assertGreaterEqual(issued, date(2010, 1, 1), document)
                self.assertLessEqual(issued, date(2025, 12, 31), document)
                for key in ("registration.date", "death_registration.date", "marriage_registration.date"):
                    if values.get(key):
                        self.assertGreaterEqual(issued, values[key], document)


class CoupleTests(unittest.TestCase):
    def test_same_sex_couples_marry_only_from_2013_05_16(self):
        for document in ("certidao_casamento", "certidao_casamento_cnj"):
            for values, record in records(document):
                sexes = record._links["marriage.couple"]
                if sexes[0] == sexes[1]:
                    self.assertGreaterEqual(values["marriage.date"], date(2013, 5, 16))

    def test_same_sex_share_is_small(self):
        from src.semantic.generators import couple_sexes
        same = sum(len(set(couple_sexes(SemanticRecord(seed, "certidao_casamento")))) == 1 for seed in range(20000))
        self.assertTrue(20 <= same <= 100, same)


class MarriageTextTests(unittest.TestCase):
    def test_old_model_fields_come_from_the_couple(self):
        for values, _ in records("certidao_casamento"):
            names = [values["spouse1.name"], values["spouse2.name"]]
            self.assertEqual(values["couple.names"], " e ".join(names))
            for number, name in zip((1, 2), names):
                self.assertIn(name, values["couple.qualification"])
                self.assertIn(values[f"spouse{number}.parents_names"], values["couple.qualification"])
            divorced = "DIVORCIADO(A)" in (values["spouse1.marital_status_before"], values["spouse2.marital_status_before"])
            self.assertEqual(divorced, "divórcio" in values["couple.documents"])
            unchanged = [values["spouse1.name_after"], values["spouse2.name_after"]] == names
            self.assertEqual(unchanged, values["couple.names_after"] is None)
            self.assertLessEqual(len(values["couple.documents"]), 300)
            self.assertLessEqual(len(values["couple.qualification"]), 600)

    def test_annotations_are_blank_or_dated_before_the_issue(self):
        import re
        for document in CERTIDOES:
            filled = 0
            for values, _ in records(document):
                for key in ("annotations.averbacoes", "annotations.voluntary"):
                    text = values.get(key)
                    if text is None:
                        continue
                    filled += 1
                    self.assertNotIn("Hospital", text)
                    self.assertLessEqual(len(text), 200)
                    for day in re.findall(r"\d{2}/\d{2}/\d{4}", text):
                        d, m, y = map(int, day.split("/"))
                        if "certificate.issue_date" in values and "pacto" not in text:
                            self.assertLessEqual(date(y, m, d), values["certificate.issue_date"], (document, text))
            self.assertGreater(filled, 0, document)


if __name__ == "__main__":
    unittest.main()
