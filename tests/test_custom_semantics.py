import random
import tempfile
import unittest
from datetime import date, time
from pathlib import Path

from src.semantic.custom import CustomGenerationUnavailable, generate_custom_value, validate_custom_definition
from src.web.jobs import JobStore
from src.web.services import ServiceError
from src.web.settings import Settings


class CustomSemanticsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.settings = Settings(storage=Path(self.temporary.name), min_free_bytes=0)
        self.store = JobStore(self.settings)
        self.owner, _ = self.store.create_workspace()
        self.other_owner, _ = self.store.create_workspace()

    def test_approved_definition_persists_and_is_scoped_to_workspace(self):
        saved = self.store.save_custom_semantic(self.owner, "custom.death.circumstance", "Circunstância do óbito",
                                                "Local informado no formulário", "text",
                                                {"type": "choices", "values": ["Domicílio", "Hospital"]})
        self.assertEqual(saved["status"], "approved")
        self.assertEqual(saved["generator"]["values"], ["Domicílio", "Hospital"])
        self.assertEqual(JobStore(self.settings).custom_semantic(self.owner, saved["key"])["id"], saved["id"])
        self.assertEqual(self.store.list_custom_semantics(self.other_owner), [])
        with self.assertRaises(ServiceError) as hidden:
            self.store.custom_semantic(self.other_owner, saved["key"])
        self.assertEqual(hidden.exception.status, 404)
        with self.assertRaises(ServiceError) as duplicate:
            self.store.save_custom_semantic(self.owner, saved["key"], "Outro", "Descrição", "text",
                                            {"type": "unavailable"})
        self.assertEqual(duplicate.exception.status, 409)
        other = self.store.save_custom_semantic(self.other_owner, saved["key"], "Outro", "Descrição", "text",
                                                {"type": "unavailable"})
        self.assertNotEqual(other["id"], saved["id"])
        updated = self.store.update_custom_semantic(self.owner, saved["key"], "Circunstância", "Descrição atualizada", "text",
                                                    {"type": "choices", "values": ["Domicílio"]})
        self.assertEqual(updated["id"], saved["id"])
        self.assertEqual(self.store.custom_semantic(self.other_owner, saved["key"])["label"], "Outro")
        self.store.delete_custom_semantic(self.owner, saved["key"])
        self.assertEqual(self.store.list_custom_semantics(self.owner), [])

    def test_generation_is_deterministic_and_typed(self):
        definitions = [
            validate_custom_definition("custom.registration.volume", "Volume", "Número do livro", "integer",
                                       {"type": "integer_range", "min": 1, "max": 999}),
            validate_custom_definition("custom.death.location", "Local", "Local declarado", "text",
                                       {"type": "choices", "values": ["Hospital", "Domicílio"]}),
            validate_custom_definition("custom.registration.verified", "Verificado", "Confirmação", "boolean",
                                       {"type": "choices", "values": [True, False]}),
            validate_custom_definition("custom.registration.date", "Data adicional", "Data informada", "date",
                                       {"type": "choices", "values": ["2025-01-31"]}),
            validate_custom_definition("custom.registration.time", "Hora adicional", "Hora informada", "time",
                                       {"type": "choices", "values": ["12:30:00"]}),
        ]
        first = [generate_custom_value(item, random.Random(42)) for item in definitions]
        second = [generate_custom_value(item, random.Random(42)) for item in definitions]
        self.assertEqual(first, second)
        self.assertEqual(first[3], date(2025, 1, 31))
        self.assertEqual(first[4], time(12, 30))

    def test_invalid_or_medical_generators_are_rejected(self):
        invalid = [
            ("birth.date", "Data", "Descrição", "date", {"type": "unavailable"}),
            ("custom.location", "Local", "Descrição", "text", {"type": "choices", "values": []}),
            ("custom.location", "Local", "Descrição", "text", {"type": "choices", "values": ["A", "A"]}),
            ("custom.location", "Local", "Descrição", "integer", {"type": "integer_range", "min": 5, "max": 2}),
            ("custom.location", "Local", "Descrição", "boolean", {"type": "choices", "values": [1]}),
            ("custom.death.cause", "Causa do óbito", "Diagnóstico", "text", {"type": "choices", "values": ["Infarto"]}),
        ]
        for arguments in invalid:
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                validate_custom_definition(*arguments)
        medical = self.store.save_custom_semantic(self.owner, "custom.death.cause", "Causa do óbito",
                                                  "Requer código validado", "text", {"type": "unavailable"})
        with self.assertRaises(CustomGenerationUnavailable):
            generate_custom_value(medical, random.Random(1))
        with self.assertRaises(ServiceError) as blocked:
            self.store.update_custom_semantic(self.owner, medical["key"], medical["label"], medical["description"],
                                              "text", {"type": "choices", "values": ["Infarto"]})
        self.assertEqual(blocked.exception.status, 422)
        self.assertEqual(self.store.custom_semantic(self.owner, medical["key"])["generator"], {"type": "unavailable"})


if __name__ == "__main__":
    unittest.main()
