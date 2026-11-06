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


if __name__ == "__main__":
    unittest.main()
